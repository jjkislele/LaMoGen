import inspect
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from thirdparties.humanml.utils.word_vectorizer import WordVectorizer
from datamodules.lbnDatasetT2C import LBNT2MDataset, collate_fn


class LBNDataT2MModule(pl.LightningDataModule):
    """DataModule for text-to-code (T2C) generation training and evaluation."""

    def __init__(self, batch_size=512, downsample=1,
                 pkl_text_train=None, pkl_lbn_train=None,
                 pkl_ids_train=None, pkl_llm_train=None,
                 pkl_text_val=None, pkl_lbn_val=None,
                 pkl_ids_val=None, pkl_llm_val=None,
                 mean_path=None, std_path=None,
                 glove_path=None,
                 num_workers=32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downsample = downsample
        self.pkl_text_train = pkl_text_train
        self.pkl_lbn_train = pkl_lbn_train
        self.pkl_ids_train = pkl_ids_train
        self.pkl_llm_train = pkl_llm_train
        self.pkl_text_val = pkl_text_val
        self.pkl_lbn_val = pkl_lbn_val
        self.pkl_ids_val = pkl_ids_val
        self.pkl_llm_val = pkl_llm_val
        self.mean_path = mean_path
        self.std_path = std_path
        # Initialize word vectorizer for T2M evaluation
        self.w_vectorizer = WordVectorizer(glove_path, 'our_vab')
        #
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = LBNT2MDataset(
            pkl_text=self.pkl_text_train,
            pkl_lbn=self.pkl_lbn_train,
            pkl_ids=self.pkl_ids_train,
            pkl_llm=self.pkl_llm_train,
            downsample=self.downsample,
            is_train=True,
        )
        self._setup_val()
        class_name = inspect.getfile(LBNT2MDataset)
        print(f"+++ [CLASS] {class_name} +++")
        print(f"+++ Datamodule: {self.pkl_lbn_train} +++")
        print(f"+++ Datamodule: {self.pkl_lbn_val} +++")
        print(f"train: {len(self.train_dataset)}, val: {len(self.val_dataset)}")

    def _setup_val(self):
        self.val_dataset = LBNT2MDataset(
            pkl_text=self.pkl_text_val,
            pkl_lbn=self.pkl_lbn_val,
            pkl_ids=self.pkl_ids_val,
            pkl_llm=self.pkl_llm_val,
            mean_path=self.mean_path,
            std_path=self.std_path,
            downsample=self.downsample,
            is_train=False,
            w_vectorizer=self.w_vectorizer,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=0,
                          batch_size=32, shuffle=True, drop_last=True,
                          collate_fn=collate_fn)
