import gzip
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path, PurePosixPath
from typing import Optional, List

import pandas as pd
import pytorch_lightning as pl
import torch
import wget
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class CovTypeDataModule(pl.LightningDataModule):
    """https://archive.ics.uci.edu/ml/datasets/covertype"""

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    FILENAME = "covtype.csv"

    LABEL_COLUMN = "Cover_type"

    BINARY_COLUMNS = [
        "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
        "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
        "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
        "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
        "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
        "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
        "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
        "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
        "Soil_Type40"
    ]
    NUMERICAL_COLUMNS = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ]

    FEATURE_COLUMNS = NUMERICAL_COLUMNS + BINARY_COLUMNS
    ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

    NUM_LABELS = 7
    NUM_FEATURES = 54

    def __init__(self, batch_size: int, num_workers: int = 4, seed: int = 0,
                 cache_dir=str(Path.home()) + "/.cache/covtype/", **kwargs):
        super(CovTypeDataModule, self).__init__()

        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.cache_dir = cache_dir
        self.kwargs = kwargs

    def prepare_data(self):
        # download if necessary
        if not Path(PurePosixPath(self.cache_dir + CovTypeDataModule.FILENAME)).exists():
            Path(PurePosixPath(self.cache_dir)).mkdir(parents=True, exist_ok=True)

            file = wget.download(CovTypeDataModule.URL, out=self.cache_dir)
            with gzip.open(file, "rb") as f_in:
                with open(self.cache_dir + CovTypeDataModule.FILENAME, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # split if necessary
        if not Path(PurePosixPath(self.cache_dir + "train_" + CovTypeDataModule.FILENAME)).exists() or \
                not Path(PurePosixPath(self.cache_dir + "val_" + CovTypeDataModule.FILENAME)).exists() or \
                not Path(PurePosixPath(self.cache_dir + "test_" + CovTypeDataModule.FILENAME)).exists():
            self._split()

    # def _preprocess(self):
    #     df = pd.read_csv(self.cache_dir + CovTypeDataModule.FILENAME, header=None, names=CovTypeDataModule.ALL_COLUMNS, index_col=False, na_filter=False)
    #
    #     pass

    def _split(self):
        """same splits as the official tabnet implementation"""
        df = pd.read_csv(self.cache_dir + CovTypeDataModule.FILENAME)

        train_val_indices, test_indices = train_test_split(range(len(df)), test_size=0.2, random_state=self.seed)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.2 / 0.6, random_state=self.seed)

        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
        train_df = train_df.sample(frac=1)

        train_df.to_csv(self.cache_dir + "train_" + CovTypeDataModule.FILENAME, index=False, header=False)
        val_df.to_csv(self.cache_dir + "val_" + CovTypeDataModule.FILENAME, index=False, header=False)
        test_df.to_csv(self.cache_dir + "test_" + CovTypeDataModule.FILENAME, index=False, header=False)

    def setup(self, stage: Optional[str] = None):
        train_df = pd.read_csv(self.cache_dir + "train_" + CovTypeDataModule.FILENAME, index_col=False, header=None, na_filter=False,
                               names=CovTypeDataModule.ALL_COLUMNS)
        val_df = pd.read_csv(self.cache_dir + "val_" + CovTypeDataModule.FILENAME, index_col=False, header=None, na_filter=False,
                             names=CovTypeDataModule.ALL_COLUMNS)
        test_df = pd.read_csv(self.cache_dir + "test_" + CovTypeDataModule.FILENAME, index_col=False, header=None, na_filter=False,
                              names=CovTypeDataModule.ALL_COLUMNS)

        # get raw features and labels
        train = train_df.to_numpy()
        X_train, y_train = train[:, :-1], train[:, -1]
        val = val_df.to_numpy()
        X_val, y_val = val[:, :-1], val[:, -1]
        test = test_df.to_numpy()
        X_test, y_test = test[:, :-1], test[:, -1]

        # zero based types/labels (1..7 => 0..6)
        y_train -= 1
        y_val -= 1
        y_test -= 1

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
        parser.add_argument("--seed", type=int, default=0, help="random seed - defaults to 0")
        parser.add_argument("--cache_dir", type=str, default=str(Path.home()) + "/.cache/covtype/",
                            help="directory to store downloaded/splitted data - defaults to ~/.cache/covtype/")

        return parent_parser
