import multiprocessing
import pickle
from argparse import Namespace
from pathlib import Path, PurePosixPath
from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from datasets.featurizer import ECFPFeaturizer, MACCSFeaturizer, ToxFeaturizer, atomic_attributions_from_mappings
from datasets.utils import split_kfold


class Hergophores:
    GROUPED = {
        "substructures_active_binding_1um": [
            "c1ccccc1CCNC",  # 1
            "c1ccccc1CN2CCCCC2",  # 2
            "c1ccccc1C(F)(F)F",  # na   # 3
        ],
        "substructures_active_binding_10um": [
            "CCOc1ccccc1",  # 4
            "c1ccccc1CNCC",  # 5
        ],
        "substructures_active_functional_1um": [
            "c1ccccc1Cc1ccccc1",  # 6
            "NCCc1ccccc1",  # 7
        ],
        "substructures_active_functional_10um": [
            "CCc1ccccc1",  # 8
            "Oc1ccccc1",  # 9
            "NCCc1ccccc1",  # 7
        ],
        "substructures_inactive_binding_1um": [
            "n1ccccc1",  # 10
            "COc1ccccc1",  # 11
            "n1ccccc1C",  # 10
            "NCCc1ccccc1",  # na #12
            "c1ccccc1C(F)(F)F",  # na # 3
        ],
        "substructures_inactive_binding_10um": [
            "n1ccccc1",  # 13
            "n1ccccc1C",  # 14
            "c1cc(C)ccc1C",  # 15
            "NCCc1ccccc1",  # na #12
            "CCNCc1ccccc1",  # 16
        ],
        "substructures_inactive_functional_1um": [
            "Oc1ccccc1",  # 17
            "CCc1ccccc1",  # 18
        ],
        "substructures_inactive_functional_10um": [
            "Cc1ccccc1",  # 19
            "CCc1ccccc1",
        ],
    }

    #
    ACTIVES_UNIQUE = [
        "CCOc1ccccc1",
        "c1ccccc1CNCC",
        "c1ccccc1CCNC",
        "c1ccccc1CN2CCCCC2",
        "c1ccccc1Cc1ccccc1"
    ]

    ACTIVES_UNIQUE_ = [
        ("CCOc1ccccc1", 1),
        ("c1ccccc1CNCC", 1),
        ("c1ccccc1CCNC", 1),
        ("c1ccccc1CN2CCCCC2", 1),
        ("c1ccccc1Cc1ccccc1", 1),
    ]

    INACTIVES_UNIQUE = [
        "n1ccccc1",
        "COc1ccccc1",
        "n1ccccc1C",
        "c1cc(C)ccc1C",
        "CCNCc1ccccc1",
    ]

    @staticmethod
    def get(
            by_smiles: Optional[List[str]] = None,
            by_groups: Optional[Union[str, List[str]]] = None,
            by_activity: Optional[int] = None,
            filter_by_group_term: str = "") -> \
            Tuple[List[str], List[int]]:

        by_groups = [*Hergophores.GROUPED] if by_groups is None else by_groups
        by_groups = [by_groups] if not isinstance(by_groups, list) else by_groups

        if by_activity == 1:
            filter_activity = "_active_"
        elif by_activity == 0:
            filter_activity = "_inactive_"
        else:
            filter_activity = ""

        hergophores, activities, tmp = [], [], []
        for k, v in Hergophores.GROUPED.items():
            if k in by_groups and filter_by_group_term in k and filter_activity in k:
                smiles = [h for h in v if h in by_smiles] if by_smiles is not None else v
                _activives = [1 if "_active_" in k else 0] * len(smiles)

                hergophores += smiles
                activities += _activives
                tmp += [h + "_" + str(a) for h, a in zip(smiles, _activives)]

        unique_indices = [tmp.index(h) for h in set(tmp)]
        hergophores, activities = zip(*[(hergophores[i], activities[i]) for i in unique_indices])

        return hergophores, activities


class HERGClassifierDataModule(pl.LightningDataModule):
    DATA_PATH = Path(PurePosixPath(__file__)).parent.parent.parent / "data/herg/data_filtered.tsv"
    IGNORE_INDEX = -100

    def __init__(self,
                 batch_size: int,
                 num_workers: int = multiprocessing.cpu_count() // 2,
                 cache_dir=str(Path.home()) + "/.cache/herg/",
                 use_cache: bool = True,
                 use_labels: List[str] = None,
                 split_type: str = "random",
                 split_size: Union[Tuple[float, float, float], Tuple[int, int, int]] = (0.6, 0.2, 0.2),
                 split_seed: int = 5180,
                 featurizer_name: str = "combined",
                 featurizer_kwargs: Optional[dict] = None,
                 standardize: bool = True,
                 featurizer_n_jobs: int = 0,
                 featurizer_mp_context: str = "fork",
                 featurizer_chunksize: int = 100,
                 **kwargs):
        """

        Args:
            batch_size: training batch size
            num_workers: num workers used for data loaders and featurizer
            cache_dir: cache directory
            use_cache: whether to use cache or not
            use_labels: which labels to use provide as targets from the herg dataset - if none is defined uses all labels (multi target)
            split_type: random or kfold
            split_size:
                If split type is random: Size of splits as tuple of floats (train size, val size, test size) - must sum to 1
                If split type is kfold: Tuple of ints (nr of folds, val fold, test fold)
            split_seed: seed for splitting
            featurizer_name: at the moment only combined is used - meaning ecfp + maccs + tox will be concatenated.
            featurizer_kwargs: featurizer kwargs for ecfp featurizer (radius, folding, return counts)
            standardize: default true - standardize features
            featurizer_mp_context: used for mp context within featurizer
        """
        super(HERGClassifierDataModule, self).__init__()

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
        self.featurizer_n_jobs = featurizer_n_jobs
        self.featurizer_mp_context = featurizer_mp_context
        self.featurizer_chunksize = featurizer_chunksize
        self.standardize = standardize

        if featurizer_name == "combined":
            self.featurizer = [
                ECFPFeaturizer(n_jobs=featurizer_n_jobs, mp_context=featurizer_mp_context, chunksize=featurizer_chunksize,
                               **featurizer_kwargs),
                MACCSFeaturizer(n_jobs=featurizer_n_jobs, mp_context=featurizer_mp_context, chunksize=featurizer_chunksize),
                ToxFeaturizer(n_jobs=featurizer_n_jobs, mp_context=featurizer_mp_context, chunksize=featurizer_chunksize)]
        else:
            raise ValueError(f"featurizer {featurizer_name} is unknown")

        self.featurizer_atomic_mappings = None

        self.kwargs = kwargs

    def prepare_data(self):
        """create and cache descriptors using defined featurizer if not already done/existing"""

        if self.featurizer_name == "combined":
            featurizer_kwargs = "_".join([k + "-" + str(v) for k, v in self.featurizer_kwargs.items()])
            cached_descriptors = self.cache_dir + "herg" + "_combined" + "_" + featurizer_kwargs + ".npz"
            cached_mappings = self.cache_dir + "herg" + "_combined" + "_" + featurizer_kwargs + ".pickle"

            if (Path(PurePosixPath(cached_descriptors)).exists() and Path(PurePosixPath(cached_mappings)).exists()) and self.use_cache:
                return

            data = pd.read_csv(HERGClassifierDataModule.DATA_PATH, sep="\t")

            # create cache dir if not existing
            Path(PurePosixPath(cached_descriptors)).parent.mkdir(parents=True, exist_ok=True)

            desc_mat = self.featurize(data["smiles"].tolist())
            featurizer_atomic_mappings = self.atomic_mappings(data["smiles"].tolist())

            sparse_desc_mat = csr_matrix(desc_mat)
            save_npz(cached_descriptors, sparse_desc_mat)

            with open(cached_mappings, "wb") as f:
                pickle.dump(featurizer_atomic_mappings, f)
        else:
            raise ValueError(f"unknown featurizer {self.featurizer_name}")

    def setup(self, stage: Optional[str] = None):
        if self.featurizer_name == "combined":
            featurizer_kwargs = "_".join([k + "-" + str(v) for k, v in self.featurizer_kwargs.items()])
            cached_descriptors = self.cache_dir + "herg" + "_combined" + "_" + featurizer_kwargs + ".npz"
            cached_mappings = self.cache_dir + "herg" + "_combined" + "_" + featurizer_kwargs + ".pickle"
        else:
            raise ValueError(f"unknown featurizer {self.featurizer_name}")

        if self.featurizer_atomic_mappings is None:
            with open(cached_mappings, "rb") as f:
                self.featurizer_atomic_mappings = pickle.load(f)

        sparse_desc_mat = load_npz(cached_descriptors)
        X = sparse_desc_mat.toarray()

        data = pd.read_csv(HERGClassifierDataModule.DATA_PATH, sep="\t")

        if self.featurizer_name == "combined" and self.featurizer_atomic_mappings is None:
            # necessary to reinitialize feature map within ecfp featurizer
            self.featurizer[0].fit_transform(data["smiles"].tolist())

        # replace nan values/not defined herg activities with an ignorable index
        y = data[self.use_labels].fillna(HERGClassifierDataModule.IGNORE_INDEX).to_numpy().astype(np.int16)

        # filter samples for which all herg activities are not defined
        indices = ~np.all((y == HERGClassifierDataModule.IGNORE_INDEX), axis=1)
        X = X[indices, ...]
        y = y[indices, ...]
        self.data = data.iloc[indices]

        # filter nan features within samples (should not be the case)
        indices = np.all(~np.isnan(X), axis=1)
        X = X[indices, ...]
        y = y[indices, ...]
        self.data = self.data.iloc[indices]

        self.input_size = X.shape[-1]

        # default split and create dataset
        self.train_indices, self.val_indices, self.test_indices = self.split_indices(len(X))
        X_train = X[self.train_indices, ...]
        y_train = y[self.train_indices, ...]
        X_val = X[self.val_indices, ...]
        y_val = y[self.val_indices, ...]
        X_test = X[self.test_indices, ...] if self.test_indices is not None else None
        y_test = y[self.test_indices, ...] if self.test_indices is not None else None

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32) if X_test is not None else None

        if self.standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test) if X_test is not None else None

        self.train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long().squeeze())
        self.val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long().squeeze())

        if X_test is not None:
            self.test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long().squeeze())
        else:
            self.test_dataset = None

    def featurize(self, smiles: List[str]) -> np.ndarray:
        """init featurizer based on provided smiles and returns featurized smiles"""

        desc_mat = []
        for featurizer in self.featurizer:
            desc_mat.append(featurizer(smiles).astype(np.uint8))

        desc_mat = np.hstack(desc_mat)

        return desc_mat

    def atomic_mappings(self, smiles: List[str]) -> List[List[List[List[Tuple[int, float]]]]]:
        featurizer_atomic_mappings = []

        for featurizer in self.featurizer:
            featurizer_atomic_mappings.append(featurizer.atomic_mappings(smiles))

        return featurizer_atomic_mappings

    # TODO - calculate class weights
    def determine_class_weights(self):
        pass

    def split_indices(self, nr_samples: int) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:

        if self.split_type == "random":
            indices = np.arange(nr_samples)

            test_size = sum(self.split_size[1:])
            train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=self.split_seed)

            if self.split_size[-1] > 0 and len(self.split_size) == 3:
                test_size = self.split_size[-1] / test_size
                val_indices, test_indices = train_test_split(val_indices, test_size=test_size, random_state=self.split_seed)
            else:
                test_indices = None

            return train_indices, val_indices, test_indices
        if self.split_type == "random_kfold":
            return split_kfold(nr_samples, self.split_size, seed=self.split_seed)
        else:
            raise ValueError(f"split type {self.split_type} not supported yet")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), num_workers=self.num_workers, pin_memory=True)
        else:
            return None

    @rank_zero_only
    def atomic_attributions(self,
                            smiles_or_mappings: Union[List[str], List[List[List[Tuple[int, int]]]]],
                            feature_attributions: np.ndarray,
                            postprocess: Optional[str] = None
                            ) -> List[np.ndarray]:

        is_smiles = True if isinstance(smiles_or_mappings[0], str) else False
        n_samples = len(smiles_or_mappings) if is_smiles else len(smiles_or_mappings[0])

        featurizer_atomic_attributions = []
        if is_smiles:

            ptr = 0
            for featurizer in self.featurizer:
                _fa = feature_attributions[:, ptr:ptr + featurizer.n_features]

                featurizer_atomic_attributions.append(featurizer.atomic_attributions(smiles_or_mappings, _fa))
                ptr += featurizer.n_features
        else:

            ptr = 0
            for atomic_mappings, featurizer in zip(smiles_or_mappings, self.featurizer):
                _fa = feature_attributions[:, ptr:ptr + featurizer.n_features]

                featurizer_atomic_attributions.append(atomic_attributions_from_mappings(
                    atomic_mappings, _fa,
                    n_jobs=self.featurizer_n_jobs,
                    mp_context=self.featurizer_mp_context,
                    chunksize=self.featurizer_chunksize,
                ))
                ptr += featurizer.n_features

        # if we have attributions from multiple featurizers summarize them
        atomic_attributions = []
        for i in range(n_samples):

            _aa = None
            for _faa in featurizer_atomic_attributions:
                _aa = _aa + _faa[i] if _aa is not None else _faa[i]

            # TODO rework - make sure postprocess handles 0/nan in denum
            if postprocess == "normalize":
                _aa = (_aa - np.min(_aa)) / (
                        np.max(_aa) - np.min(_aa))
            elif postprocess == "standardize":
                _aa = (_aa - np.mean(_aa)) / np.std(_aa)

            atomic_attributions.append(_aa)

        return atomic_attributions

    @property
    def num_classes(self) -> Union[int, List[int]]:
        if len(self.use_labels) > 1:
            return [2] * len(self.use_labels)
        else:
            return 2

    @rank_zero_only
    def log_hyperparameters(self, logger: LightningLoggerBase, ignore_param: List[str] = None, types: List = None):
        if types is None:
            types = [int, float, str, dict, list, bool, tuple]

        if ignore_param is None:
            ignore_param = ["class_weights", "featurizer_atomic_mappings", "featurizer", "track_metrics", "use_labels"]

        params = {}
        for k, v in self.__dict__.items():
            if k not in ignore_param and not k.startswith("_"):
                if type(v) in types:
                    params[k] = v

        params = Namespace(**params)

        logger.log_hyperparams(params)
