import torch
import pytorch_lightning as pl
import inspect
import numpy as np

from network.hml3d_losses import ReConsLoss
from utils.lbn_utils.lbn_codebook import codebook_size
from network.decoder import LbnDecoderAttn

from thirdparties.humanml.load_encoders import get_movement_enc, get_motion_enc
from thirdparties.humanml.utils.metrics import calculate_activation_statistics, calculate_frechet_distance

loss_fn_l1 = ReConsLoss(recons_loss='l1_smooth', nb_joints=22)
loss_fn_l2 = ReConsLoss(recons_loss='l2', nb_joints=22)


class LbnModel(pl.LightningModule):
    """LBN Codec model with T2M metric evaluation (FID)."""

    def __init__(self, dec_type='pae', loss_type='l1', batch_size=512,
                 pose_dim=263, loss_vel=0.5,
                 fid_ckpt_path=None, eval_pose_dim=263):
        super().__init__()
        self.dec_type = dec_type
        self.class_name = inspect.getfile(LbnModel)
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.pose_dim = pose_dim
        self.save_hyperparameters()
        print(f"+++ Network: {dec_type} +++")
        print(f"+++ [CLASS] {self.class_name} +++")
        print(f"+++ [Loss] {self.loss_type} +++")
        print(f"+++ [Batch Size] {self.batch_size} +++")

        # Load T2M evaluator for FID computation
        if fid_ckpt_path is not None:
            self._load_evaluator(fid_ckpt_path, eval_pose_dim)

        # Decoder
        dec = LbnDecoderAttn(nb_code=codebook_size,
                             code_dim=512,
                             pose_dim=self.pose_dim)
        self.model = dec

        # Loss
        self.loss_vel = loss_vel
        if self.loss_type == 'l1':
            self.loss = loss_fn_l1
        elif self.loss_type == 'l2':
            self.loss = loss_fn_l2

        self.all_motion_embeddings = [[], []]

    def _load_evaluator(self, fid_ckpt_path, eval_pose_dim):
        """Load movement and motion encoders for FID evaluation."""
        print(f"Load evaluator: {fid_ckpt_path}")
        ckpt_dict = torch.load(fid_ckpt_path, map_location='cuda')
        self.movement_encoder = get_movement_enc(ckpt_dict['movement_encoder'],
                                                 tgt_dim=eval_pose_dim - 4)  # remove fc
        self.motion_encoder = get_motion_enc(ckpt_dict['motion_encoder'])
        self.movement_encoder = self.movement_encoder.cuda()
        self.motion_encoder = self.motion_encoder.cuda()

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

    def shared_step(self, batch):
        code_indices = batch[8]
        gt_mask = batch[9]
        gt_mask = gt_mask.bool()
        code_indices = code_indices.float()
        pred_motion = self.model(code_indices=code_indices, masks=gt_mask)
        return pred_motion

    def loss_fn(self, batch, pred_motion):
        gt_motion = batch[4]
        gt_mask = batch[9]
        # mask
        gt_motion = gt_motion * gt_mask.unsqueeze(-1)
        pred_motion = pred_motion * gt_mask.unsqueeze(-1)
        #
        loss_motion = self.loss(pred_motion, gt_motion)
        loss_vel = self.loss.forward_vel(pred_motion, gt_motion)
        #
        total_loss = loss_motion + self.loss_vel * loss_vel
        loss_stats = {'rec': loss_motion, 'rec_v': loss_vel}
        return total_loss, loss_stats

    def better_log(self, loss_stats, stage):
        for k, v in loss_stats.items():
            if torch.is_tensor(v):
                v = v.detach()
            self.log(f"{stage}_{k}", v, sync_dist=True, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        total_loss, loss_stats = self.loss_fn(batch, output)
        # Error if NaN
        if torch.isnan(total_loss):
            raise ValueError('Loss is NaN')
        # log here
        self.log('loss', total_loss.detach(),
                 on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True,
                 batch_size=self.batch_size)
        self.better_log(loss_stats, 'train')
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        total_loss, loss_stats = self.loss_fn(batch, output)
        # Collect t2m embeddings
        self.t2m_eval_step(batch, output)
        # log here
        self.better_log(loss_stats, 'val')
        self.log('val_loss', total_loss.detach(), sync_dist=True, batch_size=self.batch_size)
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
            self.log('val_fid', fid, sync_dist=True, batch_size=self.batch_size)
        except ValueError as e:
            print(f"Error encountered: {e}")
            self.log('val_fid', 999, sync_dist=True, batch_size=self.batch_size)
        # Clear memory for each validation epoch
        self.all_motion_embeddings.clear()
        self.all_motion_embeddings = [[], []]

    def t2m_eval_step(self, batch, recs):
        word_embeddings, pos_one_hots, _, sent_lens, gts, gt_lens, _, _, _, gt_mask = batch
        # mask
        gt_motion = gts * gt_mask.unsqueeze(-1)
        pred_motion = recs * gt_mask.unsqueeze(-1)
        #
        for idx, (motions, m_lens) in enumerate([[gt_motion, gt_lens], [pred_motion, gt_lens]]):
            motion_embeddings = self._get_co_embeddings(motions=motions, m_lens=m_lens)
            self.all_motion_embeddings[idx].append(motion_embeddings.cpu().numpy())
