# export
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_extraction import DictVectorizer

import numpy as np

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


class ECFC_featurizer():
    def __init__(self, radius=6, min_fragm_occur=50, useChirality=True, useFeatures=False):
        self.v = DictVectorizer(sparse=True, dtype=np.uint16)
        self.min_fragm_occur = min_fragm_occur
        self.idx_col = None
        self.radius = radius
        self.useChirality = useChirality
        self.useFeatures = useFeatures

    def compute_fp_list(self, smiles_list):
        fp_list = []
        for smiles in smiles_list:
            try:
                if isinstance(smiles, list):
                    smiles = smiles[0]
                mol = Chem.MolFromSmiles(smiles)  # TODO small hack only applicable here!!!
                fp_list.append(AllChem.GetMorganFingerprint(mol, self.radius, useChirality=self.useChirality,
                                                            useFeatures=self.useFeatures).GetNonzeroElements())  # returns a dict
            except:
                fp_list.append({})
        return fp_list

    def fit(self, x_train):
        fp_list = self.compute_fp_list(x_train)
        Xraw = self.v.fit_transform(fp_list)
        idx_col = np.array((Xraw > 0).sum(axis=0) >= self.min_fragm_occur).flatten()
        self.idx_col = idx_col
        return Xraw[:, self.idx_col].toarray()

    def transform(self, x_test):
        fp_list = self.compute_fp_list(x_test)
        X_raw = self.v.transform(fp_list)
        return X_raw[:, self.idx_col].toarray()


if __name__ == "__main__":
    name = "bbbp"
    use_cache = True
    seed = 1

    tasks, all_dataset, transformers = dataset_loading_functions[name](featurizer="Raw", splitter=None, reload=use_cache)
    featurizer = ECFC_featurizer()

    X = all_dataset[0].ids

    X_train, X_test = train_test_split(X, test_size=.2, random_state=seed)

    X_train_f0 = featurizer.fit(X_train)
    X_test_f0 = featurizer.transform(X_test)

    X_f = featurizer.transform(X)
    X_train_f, X_test_f = train_test_split(X_f, test_size=.2, random_state=seed)

    assert np.array_equal(X_train_f0, X_train_f)
    assert np.array_equal(X_test_f0, X_test_f)

    print("abc")
