import gzip
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path, PurePosixPath
from typing import Optional, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import wget
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class FirewallDataModule(pl.LightningDataModule):
    """https://archive.ics.uci.edu/ml/datasets/Internet+Firewall+Data#"""

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00542/log2.csv"
    FILENAME = "log2.csv"

    LABELS = ["allow", "deny", "drop", "reset-both"]
    LABEL_COLUMN = "Action"

    FEATURE_COLUMNS = [
        "Source Port",
        "Destination Port",
        "NAT Source Port",
        "NAT Destination Port",
        "Bytes",
        "Bytes Sent",
        "Bytes Received",
        "Packets",
        "Elapsed Time (sec)",
        "pkts_sent",
        "pkts_received",
    ]

    ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

    NUM_LABELS = 4
    NUM_FEATURES = 11

    def __init__(self,
                 batch_size: int,
                 num_workers: int = 4,
                 split_type: str = "random",
                 split_size: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 split_seed: int = 5180,
                 cache_dir=str(Path.home()) + "/.cache/firewall/", **kwargs):
        super(FirewallDataModule, self).__init__()

        self.split_type = split_type
        self.split_size = split_size
        self.split_seed = split_seed

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.cache_dir = cache_dir
        self.kwargs = kwargs

    def prepare_data(self):
        # download if necessary
        if not Path(PurePosixPath(self.cache_dir + FirewallDataModule.FILENAME)).exists():
            Path(PurePosixPath(self.cache_dir)).mkdir(parents=True, exist_ok=True)

            wget.download(FirewallDataModule.URL, out=self.cache_dir)

        # split if necessary
        if not Path(PurePosixPath(self.cache_dir + "train_" + FirewallDataModule.FILENAME)).exists() or \
                not Path(PurePosixPath(self.cache_dir + "val_" + FirewallDataModule.FILENAME)).exists() or \
                not Path(PurePosixPath(self.cache_dir + "test_" + FirewallDataModule.FILENAME)).exists():
            self._split()

    def _split(self):
        df = pd.read_csv(self.cache_dir + FirewallDataModule.FILENAME, index_col=False, usecols=FirewallDataModule.ALL_COLUMNS)[FirewallDataModule.ALL_COLUMNS]
        df[FirewallDataModule.LABEL_COLUMN].replace(to_replace=FirewallDataModule.LABELS,
                                                         value=list(range(len(FirewallDataModule.LABELS))), inplace=True)

        test_size = sum(self.split_size[1:])
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=self.split_seed,
                                            stratify=df[FirewallDataModule.LABEL_COLUMN] if self.split_type == "stratified" else None)

        test_size = self.split_size[-1] / test_size
        val_df, test_df = train_test_split(val_df, test_size=test_size, random_state=self.split_seed,
                                           stratify=df[FirewallDataModule.LABEL_COLUMN] if self.split_type == "stratified" else None)

        train_df.to_csv(self.cache_dir + "train_" + FirewallDataModule.FILENAME, index=False, header=False)
        val_df.to_csv(self.cache_dir + "val_" + FirewallDataModule.FILENAME, index=False, header=False)
        test_df.to_csv(self.cache_dir + "test_" + FirewallDataModule.FILENAME, index=False, header=False)

    def setup(self, stage: Optional[str] = None):
        train_df = pd.read_csv(self.cache_dir + "train_" + FirewallDataModule.FILENAME, index_col=False, header=None, na_filter=False,
                               names=FirewallDataModule.ALL_COLUMNS)
        val_df = pd.read_csv(self.cache_dir + "val_" + FirewallDataModule.FILENAME, index_col=False, header=None, na_filter=False,
                             names=FirewallDataModule.ALL_COLUMNS)
        test_df = pd.read_csv(self.cache_dir + "test_" + FirewallDataModule.FILENAME, index_col=False, header=None, na_filter=False,
                              names=FirewallDataModule.ALL_COLUMNS)

        # get raw features and labels
        train = train_df.to_numpy()
        X_train, y_train = train[:, :-1], train[:, -1]
        val = val_df.to_numpy()
        X_val, y_val = val[:, :-1], val[:, -1]
        test = test_df.to_numpy()
        X_test, y_test = test[:, :-1], test[:, -1]

        self.input_size = X_train.shape[-1]

        # define tensor datasets
        self.train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long().squeeze())
        self.val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long().squeeze())
        self.test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long().squeeze())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), num_workers=self.num_workers, pin_memory=True)

    @rank_zero_only
    def log_hyperparameters(self, logger: LightningLoggerBase, ignore_param: List[str] = None, types: List = None):
        if types is None:
            types = [int, float, str, dict, list, bool, tuple]

        if ignore_param is None:
            ignore_param = ["class_weights"]

        params = {}
        for k, v in self.__dict__.items():
            if k not in ignore_param and not k.startswith("_"):
                if type(v) in types:
                    params[k] = v

        params = Namespace(**params)

        logger.log_hyperparams(params)

    @staticmethod
    def add_data_module_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("CovTypeDataModule")

        parser.add_argument("--batch_size", type=int, default=32,
                            help="batch size used for training, validation and testing - defaults to 32")
        parser.add_argument("--num_workers", type=int, default=4, help="num workers for data loaders - defaults to 4")
        parser.add_argument("--cache_dir", type=str, default=str(Path.home()) + "/.cache/covtype/",
                            help="directory to store downloaded/splitted data - defaults to ~/.cache/covtype/")

        return parent_parser


if __name__ == "__main__":
    dm = FirewallDataModule(batch_size=100)
    dm.prepare_data()
    dm.setup()
