import pickle
import torch.utils.data as data
import numpy as np
import torch
import random


# 06/24 @ [FIX] only mask during training
# 06/27 @ use LLM compose data for training/testing, add downsample ratio
# 07/01 @ compute FID each validation epoch


def collate_fn(batch):
    """Collate function for T2C (text-to-code) dataset batches."""
    is_val = False if batch[0][0] is None else True
    # Sort by text length if text info is available
    if is_val:
        batch.sort(key=lambda x: x[3], reverse=True)  # sent_len needs descending order
    word_embeddings, pos_one_hots, caption, sent_len, tokens, name, motion, m_length, mask, text_embs, lbn_vectors, lbn_lengths, lbn_llm_vectors = [], [], [], [], [], [], [], [], [], [], [], [], []
    for b in batch:
        if b is None:
            continue
        w_emb, pos, cap, s_len, tok, nam, mot, m_len, m_mask, text_emb, lbn_vector, lbn_len, lbn_llm = b
        # T2M
        if is_val:
            word_embeddings.append(torch.tensor(w_emb).float())
            pos_one_hots.append(torch.tensor(pos).float())
            motion.append(torch.tensor(mot).float())
            m_length.append(torch.tensor(m_len).float())
        else:
            word_embeddings.append(w_emb)
            pos_one_hots.append(pos)
            motion.append(mot)
            m_length.append(m_len)
        caption.append(cap)
        sent_len.append(s_len)
        tokens.append(tok)
        name.append(nam)
        mask.append(m_mask)
        # T2C
        text_embs.append(torch.tensor(text_emb).float())
        lbn_vectors.append(torch.tensor(lbn_vector).float())
        lbn_llm_vectors.append(torch.tensor(lbn_llm).float())
        lbn_lengths.append(lbn_len)
    if is_val:
        motion = torch.utils.data._utils.collate.default_collate(motion)
        m_length = torch.utils.data._utils.collate.default_collate(m_length)
        mask = torch.utils.data._utils.collate.default_collate(mask)
    text_embs = torch.utils.data._utils.collate.default_collate(text_embs)
    lbn_vectors = torch.utils.data._utils.collate.default_collate(lbn_vectors)
    lbn_llm_vectors = torch.utils.data._utils.collate.default_collate(lbn_llm_vectors)
    return word_embeddings, pos_one_hots, caption, sent_len, tokens, name, motion, m_length, mask, text_embs, lbn_vectors, lbn_lengths, lbn_llm_vectors


class LBNT2MDataset(data.Dataset):
    """Dataset for text-to-code (T2C) generation with LBN codes."""

    def __init__(self, pkl_text, pkl_ids, pkl_lbn, pkl_llm,
                 mean_path=None, std_path=None,
                 max_text_len=20, max_frame_num=200,
                 w_vectorizer=None, downsample=1,
                 is_train=True):
        self.is_train = is_train
        # Load text embeddings (note: ids order matches lbn_vec order)
        if isinstance(pkl_text, list):
            with open(pkl_text[0], 'rb') as f:
                text_embs = pickle.load(f)
            with open(pkl_text[1], 'rb') as f:
                como_embs = pickle.load(f)
                text_embs_new = {}
                for k, v in text_embs.items():
                    text_embs_new[k] = np.concatenate([v, como_embs[k]], axis=0)
                text_embs = text_embs_new
                del text_embs_new, como_embs
        else:
            with open(pkl_text, 'rb') as f:
                text_embs = pickle.load(f)
        with open(pkl_ids, 'rb') as f:
            pkl_data = pickle.load(f)
            ids = pkl_data['ids']
            # For validation, extract gt motion and mask
            if not self.is_train:
                self.txts = pkl_data['txts']
                self.tkns = pkl_data['tkns']
                self.feats = pkl_data['feats']
            else:
                self.txts, self.tkns, self.feats = None, None, None
            del pkl_data
        with open(pkl_lbn, 'rb') as f:
            lbns = pickle.load(f)
        with open(pkl_llm, 'rb') as f:
            llm = pickle.load(f)
        text_embs_reorder = [text_embs[k] for k in ids]
        llm_reorder = [llm[k] for k in ids]
        self.text_embs = text_embs_reorder
        self.lbns = lbns
        self.ids = ids
        self.llm = llm_reorder
        self.length = len(self.lbns)

        self.mean = np.load(mean_path) if mean_path else np.zeros(263)
        self.std = np.load(std_path) if std_path else np.ones(263)
        self.max_text_len = max_text_len
        self.max_frame_num = max_frame_num
        self.w_vectorizer = w_vectorizer
        self.downsample = downsample

    def __len__(self):
        return self.length

    def process_tokens(self, tokens):
        if len(tokens) < self.max_text_len:
            # Pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # Crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        return pos_one_hots, word_embeddings, sent_len

    def __getitem__(self, idx):
        idx = idx % self.length

        name = self.ids[idx]
        text_emb = self.text_embs[idx]
        # Add CoMo text embeddings
        if len(text_emb) > 1:
            keyword_indices = np.arange(0, 55, 5) + np.random.randint(0, 5, size=(11,))
            clip_emb = text_emb[[0]]
            como_emb = text_emb[1:][keyword_indices]
            # Mask during training
            if self.is_train:
                mask = np.random.binomial(n=1, p=0.5, size=11).astype(bool)
                como_emb[mask] = 0
            text_emb = np.concatenate([clip_emb, como_emb], axis=0)

        # Downsample
        lbn_vector = self.lbns[idx]
        lbn_vector = lbn_vector[::self.downsample]
        lbn_llm = self.llm[idx]
        lbn_llm = lbn_llm[::self.downsample]
        max_frame_num = self.max_frame_num // self.downsample

        # Add [EOS] since downsample is applied
        lbn_vector, lbn_vector_length = self.add_eos(lbn_vector, max_frame_num)
        lbn_llm, lbn_llm_length = self.add_eos(lbn_llm, max_frame_num)

        # Pad
        if max_frame_num > lbn_vector_length:
            lbn_vector = np.concatenate(
                (lbn_vector, np.ones((max_frame_num - lbn_vector_length, lbn_vector.shape[1])) * 2), axis=0
            )
        if max_frame_num > lbn_llm_length:
            lbn_llm = np.concatenate(
                (
                lbn_llm, np.zeros((lbn_vector_length - lbn_llm_length, lbn_vector.shape[1])),  # Pad to lbn length first
                np.ones((max_frame_num - lbn_llm_length, lbn_vector.shape[1])) * 2), axis=0
            )

        # For validation, compute T2M metric
        if not self.is_train:
            txts = self.txts[idx]
            tkns = self.tkns[idx]
            caption, bgn, end = random.choice(txts)
            tokens = random.choice(tkns)
            gt_motion = self.feats[idx]
            gt_length = len(gt_motion)
            if bgn == end and bgn == 0:
                end = gt_length
            end = int(min(end, gt_length))
            bgn = int(max(0, bgn))
            # Prepare motion
            gt_motion = gt_motion[bgn:end + 1]
            gt_length = len(gt_motion)
            # Z Normalization
            gt_motion = (gt_motion - self.mean) / self.std
            # Mask
            gt_mask = np.zeros(self.max_frame_num)
            gt_mask[:gt_length] = 1
            # Pad
            if self.max_frame_num > gt_length:
                gt_motion = np.concatenate((gt_motion,
                                            np.zeros((self.max_frame_num - gt_length, gt_motion.shape[1]))), axis=0)
            # Prepare text
            pos_one_hots, word_embeddings, sent_len = self.process_tokens(tokens)

            return (word_embeddings, pos_one_hots, caption, sent_len, '_'.join(tokens), name,
                    gt_motion, gt_length, gt_mask,
                    text_emb, lbn_vector, lbn_vector_length, lbn_llm)
        else:
            return (None, None, None, None, None, name,
                    None, None, None,
                    text_emb, lbn_vector, lbn_vector_length, lbn_llm)

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
