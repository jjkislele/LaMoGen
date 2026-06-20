import argparse
import os
from pathlib import Path

import numpy as np
import pickle
import torch
import torch.utils.data as data
from einops import repeat, rearrange
from pytorch_lightning import seed_everything

from network.lbnCodeGen import LbnGenModel as Generator
from network.lbnCodec import LbnModel as Decoder


def load_cfg(cfg_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


class LBNT2MDataset(data.Dataset):
    """Dataset for LLM compose evaluation with LBN codes."""

    def __init__(self, pkl_text, pkl_llm, max_frame_num=200):
        self.max_frame_num = max_frame_num
        # Load text embeddings
        with open(pkl_text, 'rb') as f:
            text_embs = pickle.load(f)
        with open(pkl_llm, 'rb') as f:
            llm = pickle.load(f)
        # Build indexed list
        self.ids = []
        self.llm = []
        self.text_embs = []
        for k, v in llm.items():
            self.ids.append(k)
            self.llm.append(v)
            self.text_embs.append(text_embs[k])
        self.length = len(self.llm)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        name = self.ids[idx]
        text_emb = self.text_embs[idx]
        # Add CoMo text embeddings
        if len(text_emb) > 1:
            keyword_indices = np.arange(0, 55, 5) + np.random.randint(0, 5, size=(11,))
            clip_emb = text_emb[[0]]
            como_emb = text_emb[1:][keyword_indices]
            text_emb = np.concatenate([clip_emb, como_emb], axis=0)
        # Add [EOS]
        lbn_llm = self.llm[idx]
        lbn_llm, lbn_llm_length = self.add_eos(lbn_llm, self.max_frame_num)
        # Pad
        if self.max_frame_num > lbn_llm_length:
            lbn_llm = np.concatenate(
                (lbn_llm, np.ones((self.max_frame_num - lbn_llm_length, lbn_llm.shape[1])) * 2), axis=0
            )
        return name, text_emb, lbn_llm, lbn_llm_length

    @staticmethod
    def add_eos(lbn_codebook, max_frame_num):
        lbn_length = len(lbn_codebook)
        new_lbn_length = min(lbn_length, max_frame_num - 1)
        # Add [EOS] dimension and frame
        new_lbn_code = np.zeros((new_lbn_length + 1, 158 + 1), dtype=int)
        new_lbn_code[:new_lbn_length, :158] = lbn_codebook[:new_lbn_length]
        new_lbn_code[-1, 158] = 1  # [EOS] active
        T = len(new_lbn_code)  # T includes [EOS]
        return new_lbn_code, T

    def get_loader(self, batch_size=1, shuffle=False, num_workers=0):
        return data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def decode_batch(in_digits, in_length, downsample_ratio, decoder, device):
    """Decode LBN codes to motion features.

    Args:
        in_digits: (B, T, 158) LBN codes
        in_length: list of lengths
        downsample_ratio: upsample factor
        decoder: codec decoder model
        device: torch device
    """
    B, T, d = in_digits.shape
    # Upsample
    in_digits_ds = repeat(in_digits, 'b f d -> b f n d', n=downsample_ratio)
    in_digits_ds = rearrange(in_digits_ds, 'b f n d -> b (f n) d')
    # Prepare decode mask
    out_length = []
    in_mask = torch.zeros((B, T * downsample_ratio)).bool().to(device)
    for b in range(B):
        # Remove EOS
        cur_length = (in_length[b] - 1) * downsample_ratio
        in_mask[b, :cur_length] = True
        out_length.append(cur_length)

    out_feats = decoder.model(code_indices=in_digits_ds, masks=in_mask)
    out_feats = out_feats.detach().cpu().numpy()
    return out_feats, out_length


def main():
    parser = argparse.ArgumentParser(description='LLM compose to motion generation')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Config file path, relative to the cfgs/eval/ directory, '
                             'e.g. t2m_llm_compose.yaml')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    args = parser.parse_args()

    # Resolve cfg path
    cfg_filename = args.cfg
    if os.path.isabs(cfg_filename):
        cfg_path = cfg_filename
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(project_root, 'cfgs', 'eval', cfg_filename)

    assert os.path.exists(cfg_path), f"Config file not found: {cfg_path}"
    cfg = load_cfg(cfg_path)

    print(f"+++ Using config: {cfg_path} +++")
    import yaml
    print(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))

    # Seed
    seed = cfg.get('seed', 2333)
    seed_everything(seed, workers=True)

    # Extract config
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    output_path = Path(model_cfg['generator_path']).parent / f"{Path(model_cfg['generator_path']).stem}_t2c"
    output_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Dataset
    my_dataset = LBNT2MDataset(
        pkl_text=data_cfg['pkl_text'],
        pkl_llm=data_cfg['pkl_llm'],
        max_frame_num=model_cfg.get('max_frame_num', 200),
    )
    my_dataloader = my_dataset.get_loader(batch_size=data_cfg.get('batch_size', 32), num_workers=0)

    # Load generator and decoder
    generator = Generator.load_from_checkpoint(
        checkpoint_path=model_cfg['generator_path'],
        map_location=device,
    )
    generator = generator.eval().to(device)
    decoder = Decoder.load_from_checkpoint(
        checkpoint_path=model_cfg['decoder_path'],
        map_location=device,
    )
    decoder = decoder.eval().to(device)

    # Inference loop
    idx = 0
    for batch in my_dataloader:
        name_b, text_emb_b, lbn_llm_vector_b, length_b = batch
        text_emb_b = text_emb_b.to(device).float()
        lbn_llm_vector_b = lbn_llm_vector_b.to(device).float()
        with torch.no_grad():
            pred_logits_batch = generator.sample_lbn_rt([name_b, text_emb_b, None, length_b, lbn_llm_vector_b])
            pred_logits_batch = generator.unfold(pred_logits_batch)
        # Decode to motion features
        motion_batch, length_batch = decode_batch(
            pred_logits_batch, length_b,
            downsample_ratio=model_cfg.get('downsample_ratio', 1),
            decoder=decoder, device=device,
        )
        # Save results
        for name, mot, length in zip(name_b, motion_batch, length_batch):
            output_filename = str(output_path / f"{name}.npy")
            mot = mot[:length]
            np.save(output_filename, mot)
            idx += 1
            print(f"=> {idx}: {output_filename}, {length}")


if __name__ == "__main__":
    main()
