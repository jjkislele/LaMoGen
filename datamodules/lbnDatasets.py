import pickle
import torch.utils.data as data
import numpy as np
import random


class LBNDataset(data.Dataset):
    def __init__(self, pkl_feat, pkl_lbn, **kwargs):
        with open(pkl_feat, 'rb') as f:
            pkl_data = pickle.load(f)
        self.names = pkl_data['ids']
        self.txts = pkl_data['txts']
        self.tkns = pkl_data['tkns']
        self.feats = pkl_data['feats']
        del pkl_data
        with open(pkl_lbn, 'rb') as f:
            self.lbns = pickle.load(f)
        self.length = len(self.lbns)

        self.mean = np.load(kwargs.get('mean_path', None))
        self.std = np.load(kwargs.get('std_path', None))
        self.max_text_len = kwargs.get('max_text_len', 20)
        self.max_frame_num = kwargs.get('max_frame_num', 200)
        self.w_vectorizer = kwargs.get('w_vectorizer', None)

    def __len__(self):
        return self.length

    def process_tokens(self, tokens):
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
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

        name = self.names[idx]
        txts = self.txts[idx]
        tkns = self.tkns[idx]
        caption, bgn, end = random.choice(txts)
        tokens = random.choice(tkns)
        motion = self.feats[idx]
        lbn_code = self.lbns[idx]
        T = len(motion)
        if bgn == end and bgn == 0:
            end = T
        end = int(min(end, T))
        bgn = int(max(0, bgn))

        # prepare motion
        # slice
        motion = motion[bgn:end + 1]
        lbn_code = lbn_code[bgn:end + 1]
        T = len(motion)
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        # mask
        mask = np.zeros(self.max_frame_num)
        mask[:T] = 1
        # pad
        if self.max_frame_num > T:
            motion = np.concatenate((motion, np.zeros((self.max_frame_num - T, motion.shape[1]))), axis=0)
            lbn_code = np.concatenate((lbn_code, np.zeros((self.max_frame_num - T, lbn_code.shape[1]))), axis=0)

        # 06/30 @ eval with t2m
        pos_one_hots, word_embeddings, sent_len = None, None, None
        if self.w_vectorizer is not None:
            pos_one_hots, word_embeddings, sent_len = self.process_tokens(tokens)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, T, '_'.join(tokens), name, lbn_code, mask


class LBNDataset196(data.Dataset):
    def __init__(self, pkl_feat, pkl_lbn, **kwargs):
        with open(pkl_feat, 'rb') as f:
            pkl_data = pickle.load(f)
        self.names = pkl_data['ids']
        self.txts = pkl_data['txts']
        self.tkns = pkl_data['tkns']
        self.feats = pkl_data['feats']
        del pkl_data
        with open(pkl_lbn, 'rb') as f:
            self.lbns = pickle.load(f)
        self.length = len(self.lbns)
        self.unit_length = 4

        self.mean = np.load(kwargs.get('mean_path', None))
        self.std = np.load(kwargs.get('std_path', None))
        self.max_text_len = kwargs.get('max_text_len', 20)
        self.max_frame_num = kwargs.get('max_frame_num', 200)
        self.w_vectorizer = kwargs.get('w_vectorizer', None)

    def __len__(self):
        return self.length

    def process_tokens(self, tokens):
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
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

        name = self.names[idx]
        txts = self.txts[idx]
        tkns = self.tkns[idx]
        sel_text_id = random.randint(0, len(txts) - 1)
        caption, bgn, end = txts[sel_text_id]
        tokens = tkns[sel_text_id]
        motion = self.feats[idx][:self.max_frame_num]
        lbn_code = self.lbns[idx]
        m_length = len(motion)
        if bgn == end and bgn == 0:
            end = m_length
        end = int(min(end, m_length))
        bgn = int(max(0, bgn))

        # prepare motion
        # slice
        motion = motion[bgn:end + 1]
        lbn_code = lbn_code[bgn:end + 1]
        m_length = len(motion)

        coin2 = np.random.choice(['single', 'single', 'double'])
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        fIdx = random.randint(0, len(motion) - m_length)
        motion = motion[fIdx:fIdx + m_length]
        lbn_code = lbn_code[fIdx:fIdx + m_length]
        lbn_code_len = len(lbn_code)

        "Z Normalization"
        motion = (motion - self.mean) / self.std
        # mask
        mask = np.zeros(200)
        mask[:lbn_code_len] = 1
        # pad
        if self.max_frame_num > m_length:
            motion = np.concatenate((motion, np.zeros((self.max_frame_num - m_length, motion.shape[1]))), axis=0)
        if 200 > lbn_code_len:
            lbn_code = np.concatenate((lbn_code, np.zeros((200 - lbn_code_len, lbn_code.shape[1]))), axis=0)

        # 06/30 @ eval with t2m
        pos_one_hots, word_embeddings, sent_len = None, None, None
        if self.w_vectorizer is not None:
            pos_one_hots, word_embeddings, sent_len = self.process_tokens(tokens)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(
            tokens), name, lbn_code, mask


class LBNDatasetT2M196(data.Dataset):
    def __init__(self, pkl_feat, pkl_lbn, pkl_text, pkl_llm, **kwargs):
        with open(pkl_feat, 'rb') as f:
            pkl_data = pickle.load(f)
        self.names = pkl_data['ids']
        self.txts = pkl_data['txts']
        self.tkns = pkl_data['tkns']
        self.feats = pkl_data['feats']
        del pkl_data
        with open(pkl_lbn, 'rb') as f:
            self.lbns = pickle.load(f)
        self.length = len(self.lbns)
        self.unit_length = 4

        self.mean = np.load(kwargs.get('mean_path', None))
        self.std = np.load(kwargs.get('std_path', None))
        self.max_text_len = kwargs.get('max_text_len', 20)
        self.max_frame_num = kwargs.get('max_frame_num', 200)
        self.w_vectorizer = kwargs.get('w_vectorizer', None)

        # T2M part
        with open(pkl_text, 'rb') as f:
            text_embs = pickle.load(f)
        text_embs_reorder = [text_embs[k] for k in self.names]
        with open(pkl_llm, 'rb') as f:
            llm = pickle.load(f)
        llm_reorder = [llm[k] for k in self.names]
        self.text_embs = text_embs_reorder
        self.llm = llm_reorder

    def __len__(self):
        return self.length

    def process_tokens(self, tokens):
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
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

        name = self.names[idx]
        txts = self.txts[idx]
        tkns = self.tkns[idx]
        sel_text_id = random.randint(0, len(txts) - 1)
        caption, bgn, end = txts[0]
        tokens = tkns[0]
        motion = self.feats[idx]
        m_length = len(motion)
        if bgn == end and bgn == 0:
            end = m_length
        end = int(min(end, m_length))
        bgn = int(max(0, bgn))

        # prepare motion and corresponding lbn
        # slice
        motion = motion[:self.max_frame_num]
        # motion = motion[bgn:end + 1]
        #
        coin2 = np.random.choice(['single', 'single', 'double'])
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        fIdx = random.randint(0, len(motion) - m_length)
        motion = motion[fIdx:fIdx + m_length]
        m_length = len(motion)

        "Z Normalization"
        motion = (motion - self.mean) / self.std
        # pad
        if self.max_frame_num > m_length:
            motion = np.concatenate((motion, np.zeros((self.max_frame_num - m_length, motion.shape[1]))), axis=0)

        # 06/30 @ eval with t2m
        pos_one_hots, word_embeddings, sent_len = None, None, None
        if self.w_vectorizer is not None:
            pos_one_hots, word_embeddings, sent_len = self.process_tokens(tokens)

        # T2M
        lbn_llm = self.llm[idx]
        # lbn_llm = lbn_llm[bgn:end + 1]
        # lbn_llm = lbn_llm[fIdx:fIdx + m_length]
        text_emb = self.text_embs[idx]
        lbn_llm, lbn_llm_length = self.add_eos(lbn_llm, 200)
        # pad
        if 200 > lbn_llm_length:
            lbn_llm = np.concatenate(
                (lbn_llm, np.ones((200 - lbn_llm_length, lbn_llm.shape[1])) * 2), axis=0
            )
        # mask
        mask = np.zeros(200)
        mask[:lbn_llm_length - 1] = 1  # 去掉eos位

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(
            tokens), name, mask, text_emb, lbn_llm, lbn_llm_length

    @staticmethod
    def add_eos(lbn_codebook, max_frame_num):
        lbn_length = len(lbn_codebook)
        new_lbn_length = min(lbn_length, max_frame_num - 1)
        # NOTE: First n frames: LBN; extra frame added for [EOS] token in dimension.
        new_lbn_code = np.zeros((new_lbn_length + 1, 158 + 1), dtype=int)
        new_lbn_code[:new_lbn_length, :158] = lbn_codebook[:new_lbn_length]
        new_lbn_code[-1, 158] = 1  # <- [EOS] active
        T = len(new_lbn_code)
        return new_lbn_code, T
