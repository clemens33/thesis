from argparse import Namespace
from pathlib import Path, PurePosixPath
from typing import Optional, List, Tuple, Dict

import deepchem
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from datasets.utils import add_noise_features

dataset_loading_functions = {
    "bace_c": deepchem.molnet.load_bace_classification,
    "bbbp": deepchem.molnet.load_bbbp,
    "chembl": deepchem.molnet.load_chembl,
    "clearance": deepchem.molnet.load_clearance,
    "clintox": deepchem.molnet.load_clintox,
    "delaney": deepchem.molnet.load_delaney,
    "factors": deepchem.molnet.load_factors,
    "hiv": deepchem.molnet.load_hiv,
    "hopv": deepchem.molnet.load_hopv,
    "hppb": deepchem.molnet.load_hppb,
    "kaggle": deepchem.molnet.load_kaggle,
    "kinase": deepchem.molnet.load_kinase,
    "lipo": deepchem.molnet.load_lipo,
    "muv": deepchem.molnet.load_muv,
    "nci": deepchem.molnet.load_nci,
    "pcba": deepchem.molnet.load_pcba,
    # "pcba_128": deepchem.molnet.load_pcba_128,
    # "pcba_146": deepchem.molnet.load_pcba_146,
    # "pcba_2475": deepchem.molnet.load_pcba_2475,
    # "pdbbind": deepchem.molnet.load_pdbbind_grid,
    "ppb": deepchem.molnet.load_ppb,
    # "qm7": deepchem.molnet.load_qm7_from_mat,
    # "qm7b": deepchem.molnet.load_qm7b_from_mat,
    "qm8": deepchem.molnet.load_qm8,
    "qm9": deepchem.molnet.load_qm9,
    "sampl": deepchem.molnet.load_sampl,
    "sider": deepchem.molnet.load_sider,
    "thermosol": deepchem.molnet.load_thermosol,
    "tox21": deepchem.molnet.load_tox21,
    "toxcast": deepchem.molnet.load_toxcast,
    "uv": deepchem.molnet.load_uv
}


class MolNetClassifierDataModule(pl.LightningDataModule):
    _names = ["bace_c", "bbbp", "clintox", "hiv", "muv", "pcba", "pcba_146",
              "pcba_2475", "sider", "tox21", "toxcast"]

    # focus on bbbp, bace_c and tox21

    def __init__(self, name: str,
                 batch_size: int,
                 num_workers: int = 4,
                 split: str = "random",
                 split_size: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 data_seed: int = 5180,
                 cache_dir=str(Path.home()) + "/.cache/molnet/",
                 use_cache: bool = True,
                 noise_features: Optional[Dict] = None,
                 **kwargs):
        """

        Args:
            name: molnet dataset name
            batch_size: training batch size
            num_workers: num workers used for data loaders
            split: type of split
            split_size: split size
            data_seed: seed used for splitting
            cache_dir: cache directory
            use_cache: whether to use cache or not
            noise_features: whether to add noise features
            **kwargs: featurizer params - includes n_bits, radius, chirality, features
        """
        super(MolNetClassifierDataModule, self).__init__()

        assert name in MolNetClassifierDataModule._names, f"dataset {name} not in {MolNetClassifierDataModule._names}"
        assert sum(split_size) <= 1.0, f"split sizes must sum up to 1.0"
        assert split_size[0] > 0.0, f"train split size must sum greater than 0.0"

        self.name = name
        self.split = split
        self.split_size = split_size
        self.data_seed = data_seed

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.cache_dir = cache_dir
        self.use_cache = use_cache

        self.noise_features = noise_features

        self.kwargs = kwargs

    def prepare_data(self):
        """create and cache descriptors using morgan fingerprint vectors if not already done/existing"""
        n_bits = self.kwargs["n_bits"]
        radius = self.kwargs["radius"]
        chirality = self.kwargs["chirality"] if "chirality" in self.kwargs else False
        features = self.kwargs["features"] if "features" in self.kwargs else False

        tasks, all_dataset, transformers = dataset_loading_functions[self.name](featurizer="Raw", splitter=None, data_dir=self.cache_dir,
                                                                                reload=self.use_cache)

        cached_descriptors = self.cache_dir + self.name + "_ecfp" + f"_radius{str(radius)}" + f"_n_bits{str(n_bits)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"

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



    def setup(self, stage: Optional[str] = None):
        n_bits = self.kwargs["n_bits"]
        radius = self.kwargs["radius"]
        chirality = self.kwargs["chirality"] if "chirality" in self.kwargs else False
        features = self.kwargs["features"] if "features" in self.kwargs else False

        tasks, all_dataset, transformers = dataset_loading_functions[self.name](featurizer="Raw", splitter=None, data_dir=self.cache_dir,
                                                                                reload=self.use_cache)

        # load descriptors
        cached_descriptors = self.cache_dir + self.name + "_ecfp" + f"_radius{str(radius)}" + f"_n_bits{str(n_bits)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"
        sparse_desc_mat = load_npz(cached_descriptors)

        X = sparse_desc_mat.toarray()
        y = all_dataset[0].y

        # add random features/noise
        if self.noise_features:
            X, _ = add_noise_features(X,
                                      factor=self.noise_features["factor"],
                                      type=self.noise_features["type"],
                                      position=self.noise_features["position"],
                                      seed=self.data_seed)

        self.input_size = X.shape[-1]

        # get classes/weights without sorting
        indices = np.unique(y, return_index=True)[1]
        self.classes = np.concatenate([y[i] for i in sorted(indices)])
        self.class_weights = np.concatenate([all_dataset[0].w[i] for i in sorted(indices)]) if hasattr(all_dataset[0], "w") else None

        # sort classes/weights
        sorted_indices = np.argsort(self.classes)
        self.classes = self.classes[sorted_indices]
        self.class_weights = self.class_weights[sorted_indices].tolist() if self.class_weights is not None else None

        # default split and create dataset

        test_size = sum(self.split_size[1:])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=self.data_seed,
                                                          stratify=self.classes if self.split == "stratified" else None)

        test_size = self.split_size[-1] / test_size
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size, random_state=self.data_seed,
                                                        stratify=self.classes if self.split == "stratified" else None)

        self.train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long().squeeze())
        self.val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long().squeeze())
        self.test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long().squeeze())

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
