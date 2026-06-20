import torch
import pytorch_lightning as pl
import inspect
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from thirdparties.humanml.utils.word_vectorizer import WordVectorizer
from datamodules.lbnDatasets import LBNDataset


def collate_fn(batch):
    """Collate function for LBN dataset batches."""
    # Sort by text length if text info is available
    if batch[0][0] is not None:
        batch.sort(key=lambda x: x[3], reverse=True)
    word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, name, lbn_code, mask = \
        [], [], [], [], [], [], [], [], [], []
    for b in batch:
        if b is None:
            continue
        w_emb, pos, cap, s_len, mot, m_len, tok, nam, lbn, msk = b
        if w_emb is None:
            word_embeddings.append(w_emb)
            pos_one_hots.append(pos)
        else:
            word_embeddings.append(torch.tensor(w_emb).float())
            pos_one_hots.append(torch.tensor(pos).float())
        caption.append(cap)
        sent_len.append(s_len)
        motion.append(torch.tensor(mot))
        m_length.append(torch.tensor(m_len).float())
        tokens.append(tok)
        name.append(nam)
        lbn_code.append(torch.tensor(lbn).float())
        mask.append(msk)
    if w_emb is not None:
        word_embeddings = default_collate(word_embeddings)
        pos_one_hots = default_collate(pos_one_hots)
        sent_len = default_collate(sent_len)
    motion = default_collate(motion)
    m_length = default_collate(m_length)
    lbn_code = default_collate(lbn_code)
    mask = default_collate(mask)
    return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, name, lbn_code, mask


class LBNDataModule(pl.LightningDataModule):
    """DataModule for LBN codec training and evaluation."""

    def __init__(self, batch_size=512,
                 pkl_feat_train=None, pkl_lbn_train=None,
                 pkl_feat_val=None, pkl_lbn_val=None,
                 mean_path=None, std_path=None,
                 glove_path=None,
                 num_workers=32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pkl_feat_train = pkl_feat_train
        self.pkl_lbn_train = pkl_lbn_train
        self.pkl_feat_val = pkl_feat_val
        self.pkl_lbn_val = pkl_lbn_val
        self.mean_path = mean_path
        self.std_path = std_path
        # Initialize word vectorizer for T2M evaluation
        self.w_vectorizer = WordVectorizer(glove_path, 'our_vab')
        #
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = LBNDataset(
            pkl_feat=self.pkl_feat_train,
            pkl_lbn=self.pkl_lbn_train,
            mean_path=self.mean_path,
            std_path=self.std_path
        )
        self.val_dataset = LBNDataset(
            pkl_feat=self.pkl_feat_val,
            pkl_lbn=self.pkl_lbn_val,
            mean_path=self.mean_path,
            std_path=self.std_path,
            w_vectorizer=self.w_vectorizer,
        )
        class_name = inspect.getfile(LBNDataModule)
        print(f"+++ [CLASS] {class_name} +++")
        print(f"+++ Datamodule: {self.pkl_lbn_train} +++")
        print(f"+++ Datamodule: {self.pkl_lbn_val} +++")
        print(f"train: {len(self.train_dataset)}, val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=0,
                          batch_size=32, shuffle=True, drop_last=True,
                          collate_fn=collate_fn)
