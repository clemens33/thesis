# export
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_extraction import DictVectorizer

import numpy as np


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
