import torch
import pytorch_lightning as pl
import inspect
import numpy as np

from network.lbnCodec import LbnModel
from network.backbone.ComoMotionTrans import MotionTrans

from utils.lbn_utils.lbn_codebook import to_code_range, cat_num
from thirdparties.humanml.load_encoders import get_movement_enc, get_motion_enc
from thirdparties.humanml.utils.metrics import calculate_activation_statistics, calculate_frechet_distance

loss_bce = torch.nn.BCEWithLogitsLoss()


class LbnGenModel(pl.LightningModule):
    """LBN Code Generation model with T2M metric evaluation (FID)."""

    def __init__(self, batch_size=512, text_num=1, code_num=159,
                 time_num=200, downsample=4, debug=False,
                 pkeep=0.5, decay_factor=0.99, min_sampling_prob=0.1,
                 mask_ratio=None,
                 fid_ckpt_path=None, fid_pose_dim=263,
                 decoder_ckpt_path=None):
        super().__init__()
        self.class_name = inspect.getfile(LbnGenModel)
        time_num = time_num // downsample
        self.batch_size = batch_size
        self.debug = debug
        self.save_hyperparameters()
        print(f"+++ Network: +++")
        print(f"+++ [CLASS] {self.class_name} +++")
        print(f"+++ [Code Num] {code_num} +++")
        print(f"+++ [Text Num] {text_num} +++")
        print(f"+++ [Batch Size] {self.batch_size} +++")
        print(f"+++ [Downsample ratio] {downsample} +++")

        # Model
        model = MotionTrans(num_vq=code_num,
                            embed_dim=512,
                            clip_dim=512,
                            block_size=time_num + text_num,
                            num_layers=9,
                            n_head=16,
                            drop_out_rate=0.1,
                            fc_rate=4)
        self.model = model

        self.decay_factor = decay_factor
        self.min_sampling_prob = min_sampling_prob
        self.current_sampling_prob = 1.0
        self.pkeep = pkeep
        self.time_num = time_num
        self.text_offset = text_num
        self.code_num = code_num
        self.downsample = downsample
        self.mask_ratio = mask_ratio

        self.all_motion_embeddings = [[], []]

        # Load FID evaluator and codec decoder
        if fid_ckpt_path is not None and decoder_ckpt_path is not None:
            self._load_evaluator_and_decoder(fid_ckpt_path, fid_pose_dim, decoder_ckpt_path)

    def _load_evaluator_and_decoder(self, fid_ckpt_path, fid_pose_dim,
                                    decoder_ckpt_path):
        """Load movement/motion encoders and codec decoder for FID evaluation."""
        print(f"Load evaluator: {fid_ckpt_path}")
        ckpt_dict = torch.load(fid_ckpt_path, map_location='cuda')
        self.movement_encoder = get_movement_enc(ckpt_dict['movement_encoder'],
                                                 tgt_dim=fid_pose_dim - 4)  # remove fc
        self.motion_encoder = get_motion_enc(ckpt_dict['motion_encoder'])
        self.movement_encoder = self.movement_encoder.cuda()
        self.motion_encoder = self.motion_encoder.cuda()

        # Load codec decoder for FID evaluation
        print(f"Load codec decoder: {decoder_ckpt_path}")
        self.lbn_decoder = LbnModel.load_from_checkpoint(
            checkpoint_path=decoder_ckpt_path,
            map_location='cuda',
            dec_type='attn',
            fid_ckpt_path=fid_ckpt_path,
            eval_pose_dim=fid_pose_dim,
            strict=False,  # in case w/o t2m evaluator
        )
        self.lbn_decoder = self.lbn_decoder.eval().cuda()

    def _get_co_embeddings(self, motions, m_lens):
        """Compute motion embeddings for FID evaluation.

        Args:
            motions: (B, T, D) motion sequences
            m_lens: (B,) motion lengths
        """
        with torch.no_grad():
            motions = motions.detach().to('cuda:0').float()
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            # Movement Encoding
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // 4
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

    def configure_optimizers(self):
        opt_0 = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return opt_0

    def shared_step(self, batch, train_flg=True):
        _, _, _, _, _, _, _, _, _, text_feats, input_index, m_len, llm_index = batch
        text_feats = text_feats.float()
        input_index = input_index.float()
        llm_index = llm_index.float()
        # Shift for auto-regressive generation
        input_index = input_index[:, :-1]
        llm_index = llm_index[:, :-1]
        #
        if (np.random.random() >= self.current_sampling_prob and train_flg) or self.debug:
            if self.pkeep == -1:
                proba = np.random.rand(1)[0]
                mask = torch.bernoulli(proba * torch.ones_like(input_index))
            else:
                mask = torch.bernoulli(self.pkeep * torch.ones_like(input_index))

            if torch.bernoulli(torch.tensor(self.pkeep)).item() == 1:
                mask = mask.round().to(dtype=torch.int64)
                r_indices = torch.randn(input_index.shape, device=input_index.device)
                a_indices = mask * input_index + (1 - mask) * r_indices
                # Mutual exclusivity
                a_indices = self.mutual_exclusivity(a_indices, a_indices, inplace=True)
            else:
                # LLM basic with random mask
                if self.mask_ratio is None:
                    a_indices = llm_index
                else:
                    a_indices = llm_index * torch.bernoulli(
                        torch.tensor(self.mask_ratio) * torch.ones_like(llm_index))
        else:
            # Inference: use LLM composed as input
            if self.mask_ratio is None:
                a_indices = llm_index
            else:
                a_indices = llm_index * torch.bernoulli(
                    torch.tensor(self.mask_ratio) * torch.ones_like(llm_index))
        #
        prediction = self.model(a_indices, text_feats)
        prediction = prediction.contiguous()
        return prediction

    def loss_fn(self, batch, cls_pred):
        _, _, _, _, _, _, _, _, _, _, input_index, m_len, _ = batch
        target = input_index.float()
        m_tokens_len = m_len  # Length includes [EOS], no need to add 1
        B, T, d = target.shape
        loss_cls = 0.
        right_num_batch = np.zeros(2)
        for i in range(B):
            pred = cls_pred[i, self.text_offset - 1:m_tokens_len[i] + self.text_offset - 1]  # offset - 1 for auto-reg
            tgt = target[i, :m_tokens_len[i]].float()
            # Weight control for three parts
            loss_cls = loss_cls + loss_bce(pred, tgt) / B
            # Check cls correctness
            # NOTE: In dimension, first 158 are symbols, last is [EOS], computed with sigmoid
            right_num = np.zeros(2)  # symbol
            for cIdx, (bgn, end) in enumerate(to_code_range):
                # Symbol part
                if bgn < 158:
                    right_num[0] += (torch.argmax(pred[:, bgn:end], dim=-1) ==
                                     torch.argmax(tgt[:, bgn:end], dim=-1)).sum().item()
            right_num = right_num / m_tokens_len[i]  # Per-frame accuracy per part
            right_num_batch += right_num
        right_num_batch = right_num_batch / B / np.array([cat_num, 1])  # Per batch/class/frame accuracy
        #
        loss_stats = {'cls': loss_cls, 'acc_ratio': np.average(right_num_batch),
                      'acc_0_symbol': right_num_batch[0]}
        return loss_cls, loss_stats

    def better_log(self, loss_stats, stage):
        for k, v in loss_stats.items():
            if torch.is_tensor(v):
                v = v.detach()
            self.log(f"{stage}_{k}", v, sync_dist=True, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        # Adjust sampling probability
        if batch_idx > 0:
            self.current_sampling_prob = max(self.min_sampling_prob,
                                             self.current_sampling_prob * self.decay_factor)
        output = self.shared_step(batch)
        total_loss, loss_stats = self.loss_fn(batch, output)
        # Error if NaN
        if torch.isnan(total_loss):
            raise ValueError('Loss is NaN')
        # Log
        self.log('prob', self.current_sampling_prob, batch_size=self.batch_size, sync_dist=True)
        self.log('loss', total_loss.detach(), batch_size=self.batch_size,
                 on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        self.better_log(loss_stats, 'train')
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, train_flg=False)
        total_loss, loss_stats = self.loss_fn(batch, output)
        # Log
        self.better_log(loss_stats, 'val')
        self.log('val_loss', total_loss.detach(), batch_size=self.batch_size, sync_dist=True)
        # Collect T2M embeddings
        self.t2m_eval_step(batch, output)
        return total_loss

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """Compute T2M metric (FID)."""
        self.all_motion_embeddings[0] = np.concatenate(self.all_motion_embeddings[0], axis=0)
        self.all_motion_embeddings[1] = np.concatenate(self.all_motion_embeddings[1], axis=0)
        gt_mu, gt_cov = calculate_activation_statistics(self.all_motion_embeddings[0])
        mu, cov = calculate_activation_statistics(self.all_motion_embeddings[1])
        try:
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
            fid = np.abs(fid)
            self.log('val_fid', fid, sync_dist=True, batch_size=self.batch_size)
        except ValueError as e:
            print(f"Error encountered: {e}")
            self.log('val_fid', 999, sync_dist=True, batch_size=self.batch_size)
        # Clear memory for each validation epoch
        self.all_motion_embeddings.clear()
        self.all_motion_embeddings = [[], []]

    @torch.no_grad()
    def sample_lbn(self, batch):
        _, _, _, _, _, _, _, _, _, text_feats, _, _, llm_index = batch
        text_feats = text_feats.float()
        llm_index = llm_index.float()
        #
        llm_index = llm_index[:, :-1]
        #
        prediction = self.model(llm_index, text_feats)
        prediction = prediction.contiguous()
        symbs = torch.zeros_like(prediction)
        symbs = self.mutual_exclusivity(symbs, prediction, inplace=False)

        return symbs

    @torch.no_grad()
    def sample_lbn_rt(self, batch, mask_ratio=None):
        _, text_feats, _, _, llm_index = batch
        text_feats = text_feats.float()
        llm_index = llm_index.float()
        #
        llm_index = llm_index[:, :-1]
        #
        if mask_ratio is None:
            llm_index = llm_index
        else:
            llm_index = llm_index * torch.bernoulli(mask_ratio * torch.ones_like(llm_index))
        #
        prediction = self.model(llm_index, text_feats)
        prediction = prediction.contiguous()
        symbs = torch.zeros_like(prediction)
        symbs = self.mutual_exclusivity(symbs, prediction, inplace=False)

        return symbs

    @torch.no_grad()
    def t2m_eval_step(self, batch, recs):
        word_embeddings, pos_one_hots, caption, sent_lens, token, name, gts, m_length, gt_mask, text_emb, lbn_vector, lbn_length, lbn_llm_vector = batch
        # C2M: laban decoder
        pred_lbns = self.sample_lbn(batch)
        pred_lbns = self.unfold(pred_lbns)
        # Prepare decode mask
        B, T, d = pred_lbns.shape
        in_mask = torch.zeros((B, T * self.downsample)).bool().to(self.device)
        for b in range(B):
            # Remove EOS
            cur_length = (lbn_length[b] - 1) * self.downsample
            in_mask[b, :cur_length] = True
        pred_motion = self.lbn_decoder.model(code_indices=pred_lbns, masks=in_mask)
        # Mask
        gt_motion = gts * gt_mask.unsqueeze(-1)
        pred_motion = pred_motion * in_mask.unsqueeze(-1)
        #
        for idx, (motions, m_lens) in enumerate([[gt_motion, m_length], [pred_motion, m_length]]):
            motion_embeddings = self._get_co_embeddings(motions=motions, m_lens=m_lens)
            self.all_motion_embeddings[idx].append(motion_embeddings.cpu().numpy())

    @staticmethod
    def unfold(digits):
        return digits[..., :158]

    @staticmethod
    def mutual_exclusivity(symbs, probs, inplace=False):
        # Mutual exclusivity
        for cIdx, (bgn, end) in enumerate(to_code_range):
            # Symbol part
            if bgn < 158:
                idx = torch.argmax(probs[:, :, bgn:end], dim=-1, keepdim=True)
                if inplace:
                    symbs[:, :, bgn:end] = 0
                symbs.scatter_(-1, bgn + idx, 1)
            # [EOS] part
            elif bgn == 158:
                symbs[:, :, bgn:bgn + 1] = (torch.nn.functional.sigmoid(probs[:, :, bgn:bgn + 1]) > 0.5)
        return symbs
