from argparse import Namespace
from pathlib import Path, PurePosixPath
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from datasets.featurizer import ECFPFeaturizer, MACCSFeaturizer, ToxFeaturizer


class HERGClassifierDataset(pl.LightningDataModule):
    DATA_PATH = Path(PurePosixPath(__file__)).parent.parent.parent / "data/herg/data_filtered.tsv"

    def __init__(self,
                 batch_size: int,
                 num_workers: int = 4,
                 cache_dir=str(Path.home()) + "/.cache/herg/",
                 use_cache: bool = True,
                 use_labels: List[str] = None,
                 split_type: str = "random",
                 split_size: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                 split_seed: int = 5180,
                 featurizer_name: str = "combined",
                 featurizer_kwargs: Optional[dict] = None,
                 **kwargs):
        """

        Args:
            batch_size: training batch size
            num_workers: num workers used for data loaders
            cache_dir: cache directory
            use_cache: whether to use cache or not
            use_labels: which labels to use provide as targets from the herg dataset
            split_type: random or stratified
            split_size: size of split as Tuple(train size, val size, test size) - must sum to 1
            split_seed: seed for splitting
            featurizer_name: at the moment only combined is used meaning ecfp + maccs + tox will be concatenated.
            featurizer_kwargs: featurizer kwargs for ecfp featurizer (radius, folding, return counts)
        """
        super(HERGClassifierDataset, self).__init__()

        self.use_labels = use_labels if use_labels else ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80",
                                                         "active_g100"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.split_type = split_type
        self.split_size = split_size
        self.split_seed = split_seed
        self.featurizer_name = featurizer_name
        self.featurizer_kwargs = featurizer_kwargs

        if featurizer_name == "ecfp":
            self.featurizer = ECFPFeaturizer(**featurizer_kwargs)
        elif featurizer_name == "maccs":
            self.featurizer = MACCSFeaturizer()
        elif featurizer_name == "tox":
            self.featurizer = ToxFeaturizer()
        elif featurizer_name == "combined":
            self.featurizer = [ECFPFeaturizer(**featurizer_kwargs), MACCSFeaturizer(), ToxFeaturizer()]
        else:
            raise ValueError(f"featurizer {featurizer_name} is unknown")

        self.kwargs = kwargs

    def prepare_data(self):
        """create and cache descriptors using defined featurizer if not already done/existing"""

        if self.featurizer_name == "combined":
            featurizer_kwargs = "_".join([k + "-" + str(v) for k, v in self.featurizer_kwargs.items()])
            cached_descriptors = self.cache_dir + "herg" + "_combined" + "_" + featurizer_kwargs + ".npz"

            if Path(PurePosixPath(cached_descriptors)).exists() and self.use_cache:
                return

            data = pd.read_csv(HERGClassifierDataset.DATA_PATH, sep="\t")

            # create cache dir if not existing
            Path(PurePosixPath(cached_descriptors)).parent.mkdir(parents=True, exist_ok=True)

            desc_mat = []
            for featurizer in self.featurizer:
                desc_mat.append(featurizer(data["smiles"].tolist()).astype(np.uint8))

            desc_mat = np.hstack(desc_mat)

            sparse_desc_mat = csr_matrix(desc_mat)
            save_npz(cached_descriptors, sparse_desc_mat)
        else:
            raise ValueError(f"unknown featurizer {self.featurizer_name}")

    def setup(self, stage: Optional[str] = None):
        if self.featurizer_name == "combined":
            featurizer_kwargs = "_".join([k + "-" + str(v) for k, v in self.featurizer_kwargs.items()])
            cached_descriptors = self.cache_dir + "herg" + "_combined" + "_" + featurizer_kwargs + ".npz"

        else:
            raise ValueError(f"unknown featurizer {self.featurizer_name}")

        sparse_desc_mat = load_npz(cached_descriptors)
        X = sparse_desc_mat.toarray()

        data = pd.read_csv(HERGClassifierDataset.DATA_PATH, sep="\t")

        y = data[self.use_labels].fillna(-1).to_numpy().astype(np.int16)

        # filter nan samples (should not be the case)
        indices = np.all(~np.isnan(X), axis=1)
        X = X[indices, ...]
        y = y[indices]

        self.input_size = X.shape[-1]

        # default split and create dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split(X, y)

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)

        self.train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long().squeeze())
        self.val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long().squeeze())
        self.test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long().squeeze())

    def split(self, X, y):
        test_size = sum(self.split_size[1:])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=self.split_seed,
                                                          stratify=self.classes if self.split_type == "stratified" else None)

        test_size = self.split_size[-1] / test_size
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size, random_state=self.split_seed,
                                                        stratify=self.classes if self.split_type == "stratified" else None)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if len(self.test_dataset) > 0:
            return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), num_workers=self.num_workers, pin_memory=True)
        else:
            return None

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