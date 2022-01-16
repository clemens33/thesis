from argparse import Namespace
from pathlib import Path, PurePosixPath
from typing import Optional, List, Tuple, Dict

import deepchem
import numpy as np
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

from datasets.featurizer import ECFC_featurizer, ECFPFeaturizer, MACCSFeaturizer, ToxFeaturizer
from datasets.utils import add_noise_features, add_noise

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
              "pcba_2475", "sider", "tox21", "toxcast", "herg"]

    # focus on bbbp, bace_c and tox21

    def __init__(self, name: str,
                 batch_size: int,
                 num_workers: int = 4,
                 split_type: str = "random",
                 split_size: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 split_seed: int = 5180,
                 cache_dir=str(Path.home()) + "/.cache/molnet/",
                 use_cache: bool = True,
                 noise_features: Optional[Dict] = None,
                 noise: Optional[str] = None,
                 featurizer_name: str = "ecfp",
                 standardize: bool = True,
                 featurizer_n_jobs: int = 0,
                 featurizer_mp_context: str = "fork",
                 featurizer_chunksize: int = 100,
                 **kwargs):
        """

        Args:
            name: molnet dataset name
            batch_size: training batch size
            num_workers: num workers used for data loaders
            split_type: type of split
            split_size: split size
            split_seed: seed used for splitting + noise (if defined)
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
        self.split_type = split_type
        self.split_size = split_size
        self.split_seed = split_seed

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.cache_dir = cache_dir
        self.use_cache = use_cache

        self.noise_features = noise_features
        self.noise = noise

        self.featurizer_name = featurizer_name
        self.featurizer_n_jobs = featurizer_n_jobs
        self.featurizer_mp_context = featurizer_mp_context
        self.featurizer_chunksize = featurizer_chunksize

        self.kwargs = kwargs

    def prepare_data(self):
        """create and cache descriptors using morgan fingerprint vectors if not already done/existing"""

        if self.featurizer_name == "ecfp":
            n_bits = self.kwargs["n_bits"]
            radius = self.kwargs["radius"]
            chirality = self.kwargs["chirality"] if "chirality" in self.kwargs else False
            features = self.kwargs["features"] if "features" in self.kwargs else False

            splitter = "scaffold" if self.split_type == "scaffold" else None
            if splitter == "scaffold":
                raise NotImplementedError("scaffold not fully supported")

            tasks, all_dataset, transformers = dataset_loading_functions[self.name](featurizer="Raw", splitter=splitter,
                                                                                    data_dir=self.cache_dir,
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
        elif self.featurizer_name == "ecfc":
            radius = self.kwargs["radius"]
            chirality = self.kwargs["chirality"] if "chirality" in self.kwargs else False
            features = self.kwargs["features"] if "features" in self.kwargs else False

            tasks, all_dataset, transformers = dataset_loading_functions[self.name](featurizer="Raw", splitter=None,
                                                                                    data_dir=self.cache_dir,
                                                                                    reload=self.use_cache)

            cached_descriptors = self.cache_dir + self.name + "_ecfc" + f"_radius{str(radius)}" + f"_seed{str(self.split_seed)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"
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
            tasks, all_dataset, transformers = dataset_loading_functions[self.name](featurizer="Raw", splitter=None,
                                                                                    data_dir=self.cache_dir,
                                                                                    reload=self.use_cache)

            cached_descriptors = self.cache_dir + self.name + "_rdkit" + ".npz"
            if Path(PurePosixPath(cached_descriptors)).exists() and self.use_cache:
                return

            d = RDKitDescriptors()
            desc_mat = d.featurize(all_dataset[0].X)

            Path(PurePosixPath(cached_descriptors)).parent.mkdir(parents=True, exist_ok=True)
            sparse_desc_mat = csr_matrix(desc_mat)
            save_npz(cached_descriptors, sparse_desc_mat)
        elif self.featurizer_name == "combined":
            n_bits = self.kwargs["n_bits"]
            radius = self.kwargs["radius"]
            chirality = self.kwargs["chirality"] if "chirality" in self.kwargs else False
            features = self.kwargs["features"] if "features" in self.kwargs else False

            cached_descriptors = self.cache_dir + self.name + "_combined" + f"_radius{str(radius)}" + f"_n_bits{str(n_bits)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"

            if Path(PurePosixPath(cached_descriptors)).exists() and self.use_cache:
                return

            # get data
            tasks, all_dataset, transformers = dataset_loading_functions[self.name](featurizer="Raw", splitter=None,
                                                                                    data_dir=self.cache_dir,
                                                                                    reload=self.use_cache)
            # init featurizer
            self.featurizer = [
                ECFPFeaturizer(n_jobs=self.featurizer_n_jobs, mp_context=self.featurizer_mp_context, chunksize=self.featurizer_chunksize,
                               radius=radius, fold=n_bits, use_chirality=chirality, use_features=features),
                MACCSFeaturizer(n_jobs=self.featurizer_n_jobs, mp_context=self.featurizer_mp_context, chunksize=self.featurizer_chunksize),
                ToxFeaturizer(n_jobs=self.featurizer_n_jobs, mp_context=self.featurizer_mp_context, chunksize=self.featurizer_chunksize)]

            def _featurize(smiles: List[str]) -> np.ndarray:
                """init featurizer based on provided smiles and returns featurized smiles"""

                desc_mat = []
                for featurizer in self.featurizer:
                    desc_mat.append(featurizer(smiles).astype(np.uint8))

                desc_mat = np.hstack(desc_mat)

                return desc_mat

            #
            desc_mat = _featurize(all_dataset[0].ids.tolist())

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

        tasks, all_dataset, transformers = dataset_loading_functions[self.name](featurizer="Raw", splitter=None, data_dir=self.cache_dir,
                                                                                reload=self.use_cache)

        # load descriptors
        if self.featurizer_name == "ecfp":
            cached_descriptors = self.cache_dir + self.name + "_ecfp" + f"_radius{str(radius)}" + f"_n_bits{str(n_bits)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"
        elif self.featurizer_name == "ecfc":
            cached_descriptors = self.cache_dir + self.name + "_ecfc" + f"_radius{str(radius)}" + f"_seed{str(self.split_seed)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"
        elif self.featurizer_name == "rdkit":
            cached_descriptors = self.cache_dir + self.name + "_rdkit" + ".npz"
        elif self.featurizer_name == "combined":
            cached_descriptors = self.cache_dir + self.name + "_combined" + f"_radius{str(radius)}" + f"_n_bits{str(n_bits)}" + f"_chirality{str(chirality)}" + f"_features{str(features)}" + ".npz"
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

        # get classes/labels and their class weights
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
        if self.name in ["tox21", "sider"]:
            # TODO rework workaround for tox21 to proper support multi task molnet datasets

            return [2] * y.shape[-1], None

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
