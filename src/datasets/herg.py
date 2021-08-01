from argparse import Namespace
from pathlib import Path, PurePosixPath
from typing import Optional, List, Tuple, Dict

import deepchem
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from deepchem.feat import RDKitDescriptors
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from datasets.featurizer import ECFC_featurizer


class HERGClassifierDataset(pl.LightningDataModule):
    DATA_PATH = "../../data/herg/"

    def __init__(self,
                 batch_size: int,
                 num_workers: int = 4,
                 cache_dir=str(Path.home()) + "/.cache/herg/",
                 use_cache: bool = True,
                 featurizer_name: str = "ecfp",
                 featurizer_kwargs: Optional[dict] = None,
                 **kwargs):
        """

        Args:
            batch_size: training batch size
            num_workers: num workers used for data loaders
            cache_dir: cache directory
            use_cache: whether to use cache or not
            **kwargs: featurizer params - includes n_bits, radius, chirality, features
        """
        super(HERGClassifierDataset, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.featurizer_name = featurizer_name
        self.featurizer_kwargs = featurizer_kwargs

        self.kwargs = kwargs

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(HERGClassifierDataset.DATA_PATH + "train.tsv", sep="\t", dtype=str)
        validation_df = pd.read_csv(HERGClassifierDataset.DATA_PATH + "validation.tsv", sep="\t", dtype=str)
        test_df = pd.read_csv(HERGClassifierDataset.DATA_PATH + "test.tsv", sep="\t", dtype=str)

        return train_df, validation_df, test_df

    def prepare_data(self):
        """create and cache descriptors using morgan fingerprint vectors if not already done/existing"""

        if self.featurizer_name == "ecfp":
            n_bits = self.featurizer_kwargs["n_bits"]
            radius = self.featurizer_kwargs["radius"]
            chirality = self.featurizer_kwargs["chirality"] if "chirality" in self.featurizer_kwargs else False
            features = self.featurizer_kwargs["features"] if "features" in self.featurizer_kwargs else False

            cached_descriptors = self.cache_dir + "herg" + "_ecfp" + f"_radius{str(radius)}" + f"_n_bits{str(n_bits)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"

            if Path(PurePosixPath(cached_descriptors)).exists() and self.use_cache:
                return

            desc_mat = np.zeros((len(all_dataset[0].X), n_bits))

            pbar = tqdm(all_dataset[0].X)
            for i, mol in enumerate(pbar):
                fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=chirality, useFeatures=features)
                desc_mat[i] = fps

            # create cache dir if not existing
            Path(PurePosixPath(cached_descriptors)).parent.mkdir(parents=True, exist_ok=True)

            sparse_desc_mat = csr_matrix(desc_mat)
            save_npz(cached_descriptors, sparse_desc_mat)
        elif self.featurizer_name == "ecfc":
            radius = self.kwargs["radius"]
            chirality = self.kwargs["chirality"] if "chirality" in self.kwargs else False
            features = self.kwargs["features"] if "features" in self.kwargs else False



            cached_descriptors = self.cache_dir + "herg" + "_ecfc" + f"_radius{str(radius)}" + f"_seed{str(self.split_seed)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"
            if Path(PurePosixPath(cached_descriptors)).exists() and self.use_cache:
                return

            X = all_dataset[0].ids
            y = all_dataset[0].y

            (X_train, y_train), *_ = self.split(X, y)
            self.featurizer = ECFC_featurizer(radius=radius, useChirality=chirality, useFeatures=features)
            self.featurizer.fit(X_train)

            desc_mat = self.featurizer.transform(X)

            Path(PurePosixPath(cached_descriptors)).parent.mkdir(parents=True, exist_ok=True)
            sparse_desc_mat = csr_matrix(desc_mat)
            save_npz(cached_descriptors, sparse_desc_mat)
        elif self.featurizer_name == "rdkit":
            tasks, all_dataset, transformers = dataset_loading_functions["herg"](featurizer="Raw", splitter=None,
                                                                                    data_dir=self.cache_dir,
                                                                                    reload=self.use_cache)

            cached_descriptors = self.cache_dir + "herg" + "_rdkit" + ".npz"
            if Path(PurePosixPath(cached_descriptors)).exists() and self.use_cache:
                return

            d = RDKitDescriptors()
            desc_mat = d.featurize(all_dataset[0].X)

            Path(PurePosixPath(cached_descriptors)).parent.mkdir(parents=True, exist_ok=True)
            sparse_desc_mat = csr_matrix(desc_mat)
            save_npz(cached_descriptors, sparse_desc_mat)
        else:
            raise ValueError(f"unknown featurizer {self.featurizer_name}")

    def setup(self, stage: Optional[str] = None):
        n_bits = self.kwargs["n_bits"]
        radius = self.kwargs["radius"]
        chirality = self.kwargs["chirality"] if "chirality" in self.kwargs else False
        features = self.kwargs["features"] if "features" in self.kwargs else False

        tasks, all_dataset, transformers = dataset_loading_functions["herg"](featurizer="Raw", splitter=None, data_dir=self.cache_dir,
                                                                                reload=self.use_cache)

        # load descriptors
        if self.featurizer_name == "ecfp":
            cached_descriptors = self.cache_dir + "herg" + "_ecfp" + f"_radius{str(radius)}" + f"_n_bits{str(n_bits)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"
        elif self.featurizer_name == "ecfc":
            cached_descriptors = self.cache_dir + "herg" + "_ecfc" + f"_radius{str(radius)}" + f"_seed{str(self.split_seed)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"
        elif self.featurizer_name == "rdkit":
            cached_descriptors = self.cache_dir + "herg" + "_rdkit" + ".npz"
        else:
            raise ValueError(f"unknown featurizer {self.featurizer_name}")

        sparse_desc_mat = load_npz(cached_descriptors)

        X = sparse_desc_mat.toarray()
        y = all_dataset[0].y

        # filter nan samples
        indices = np.all(~np.isnan(X), axis=1)
        X = X[indices, ...]
        y = y[indices]

        if self.noise:
            X = add_noise(X, type=self.noise, seed=self.split_seed)

        # add random features/noise
        if self.noise_features:
            X, _ = add_noise_features(X,
                                      factor=self.noise_features["factor"],
                                      type=self.noise_features["type"],
                                      position=self.noise_features["position"],
                                      seed=self.split_seed)

        self.input_size = X.shape[-1]

        # automatically determine categorical variables
        X_int = X.astype(int)
        mask = np.all((X - X_int) == 0, axis=0)

        categorical_sizes = np.max(X, axis=0) + 1
        self.categorical_sizes = categorical_sizes[mask].astype(int).tolist()
        self.categorical_indices = np.argwhere(mask == True).flatten().tolist()

        w = all_dataset[0].w if hasattr(all_dataset[0], "w") else None
        self.classes, self.class_weights = self.determine_classes(y, w)

        # default split and create dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split(X, y)

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)

        self.train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long().squeeze())
        self.val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long().squeeze())
        self.test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long().squeeze())

    def determine_classes(self, y, w):
        # get classes/weights without sorting
        indices = np.unique(y, return_index=True)[1]
        classes = np.concatenate([y[i] for i in sorted(indices)])
        class_weights = np.concatenate([w[i] for i in sorted(indices)]) if w is not None else None

        # sort classes/weights
        sorted_indices = np.argsort(classes)
        classes = classes[sorted_indices]
        class_weights = class_weights[sorted_indices].tolist() if class_weights is not None else None

        return classes, class_weights

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


if __name__ == "__main__":
    dm = MolNetClassifierDataModule(
        name="bbbp",
        batch_size=128,
        split_seed=0,
        featurizer_name="rdkit",
    )

    dm.prepare_data()
    dm.setup()
