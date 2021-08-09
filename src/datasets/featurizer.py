import json
import warnings
from pathlib import PurePosixPath, Path
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, MACCSkeys
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm


class ECFC_featurizer():
    """based on the implemenation provided by P. Seidl"""

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


class ECFPFeaturizer():
    def __init__(self,
                 radius: int = 2,
                 fold: Optional[int] = None,
                 use_chirality: bool = True,
                 use_features: bool = True,
                 return_count: bool = True,
                 map_dict: Optional[dict] = None
                 ):

        self.radius = radius
        self.fold = fold
        self.use_chirality = use_chirality
        self.use_features = use_features
        self.map_dict = map_dict
        self.return_count = return_count

    @property
    def n_features(self) -> int:
        if self.fold is None:
            return len(self.map_dict) if self.map_dict else -1
        else:
            return self.fold

    def _ecfp(self, smiles: List[str]) -> Tuple[List[dict], List[dict], List[Mol]]:
        fingerprints, bit_infos, mols = [], [], []

        for idx, smile in enumerate(tqdm(smiles, desc="_ecfp")):
            mol = Chem.MolFromSmiles(smile)
            mols.append(mol)

            if mol is None:
                warnings.warn(f"could not parse smile {idx}: {smile}")

                fingerprints.append(None)
                bit_infos.append(None)
            else:
                bit_info = {}
                fingerprints.append(AllChem.GetMorganFingerprint(mol, radius=self.radius, useChirality=self.use_chirality,
                                                                 useFeatures=self.use_features, bitInfo=bit_info).GetNonzeroElements())
                bit_infos.append(bit_info)

        return fingerprints, bit_infos, mols

    def _fit(self, fingerprints: List[dict]):
        if self.map_dict is None:
            features = sorted(list(set.union(*[set(s.keys()) for s in fingerprints])))

            if self.fold is None:
                self.map_dict = dict(zip(features, range(len(features))))
            else:
                self.map_dict = {f: f % self.fold for f in features}

    def fit_transform(self, smiles: List[str]) -> np.ndarray:
        fingerprints, *_ = self._ecfp(smiles)
        self._fit(fingerprints)

        desc_mat = np.zeros((len(fingerprints), self.n_features), dtype=np.uint8)

        for i, fp in enumerate(fingerprints):
            for f, cnt in fp.items():
                if f in self.map_dict:
                    desc_mat[i, self.map_dict[f]] = cnt
                else:
                    warnings.warn(f"feature {f} not in map")

        return desc_mat

    def __call__(self, smiles: List[str]) -> np.ndarray:
        features = self.fit_transform(smiles)

        return features if self.return_count else np.where(features > 0, 1, 0).astype(features.dtype)

    def _atomic_attribution(self,
                            mol: Mol,
                            feature_attribution: np.ndarray,
                            num_atoms: Optional[int] = None,
                            bit_info: Optional[dict] = None) -> np.ndarray:
        """
        gets the individual atomic contribution for one molecule based on the feature attribution

        based and adapted from the implementation provided by J. Schimunek
        """

        num_atoms = mol.GetNumAtoms() if not num_atoms else num_atoms

        if bit_info is None:
            bit_info = {}
            AllChem.GetMorganFingerprint(mol, radius=self.radius, useChirality=self.use_chirality, useFeatures=self.use_features,
                                         bitInfo=bit_info)

        atomic_attribution = np.zeros(num_atoms)

        for f, value in bit_info.items():
            # feature mapping to account for e.g. folding
            f = self.map_dict[f]

            attribution_value = feature_attribution[f]

            for center_atom, radius in value:
                attribution_submol = np.zeros(num_atoms)

                atom_sums = 0
                if radius > 0:
                    env_mol = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                    atom_map = {}

                    Chem.PathToSubmol(mol, env_mol, atomMap=atom_map)

                    for atom_k in atom_map.keys():
                        attribution_submol[atom_k] += attribution_value
                        atom_sums += 1
                else:
                    attribution_submol[center_atom] += attribution_value
                    atom_sums = 1

                attribution_submol /= atom_sums
                atomic_attribution += attribution_submol

        return atomic_attribution

    def atomic_attributions(self, smiles: List[str], feature_attributions: np.ndarray) -> List[np.ndarray]:
        assert len(smiles) == len(
            feature_attributions), f"provided number of smiles {len(smiles)} does not match number of features {len(feature_attributions)}"

        fingerprints, bit_infos, mols = self._ecfp(smiles)

        if self.map_dict is None:
            self._fit(fingerprints)

        atomic_attributions = []
        for i, (smile, fingerprint, bit_info, mol) in enumerate(zip(smiles, fingerprints, bit_infos, mols)):
            if mol is None:
                raise ValueError(f"could not process smile/molecule {i}: {smile}")

            atomic_attribution = self._atomic_attribution(mol, feature_attributions[i], bit_info=bit_info)

            atomic_attributions.append(atomic_attribution)

        return atomic_attributions


def _smarts_substr() -> Dict[int, Mol]:
    with open(Path(PurePosixPath(__file__)).parent / "resources/maccs_smarts_substr.json") as file:
        data = json.load(file)

    return {int(k): Chem.MolFromSmarts(smile) for k, smile in data.items()}


class MACCSFeaturizer():
    SMARTS_ATOMIC_NUMBER = {
        2: [104],  # atomic num >103 Not complete, RDKit only accepts up to #104
        3: [32, 33, 34, 50, 51, 52, 82, 83, 84],  # Group IVa,Va,VIa Rows 4-6
        4: [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103],  # actinide
        5: [21, 22, 39, 40, 72],  # Group IIIB,IVB
        6: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],  # Lanthanide
        7: [23, 24, 25, 41, 42, 43, 73, 74, 75],  # Group VB,VIB,VIIB
        9: [26, 27, 28, 44, 45, 46, 76, 77, 78],  # Group VIII
        10: [4, 12, 20, 38, 56, 88],  # Group II
        12: [29, 30, 47, 48, 79, 80],  # Group IB,IIB
        18: [5, 13, 31, 49, 81],  # Group III
        27: [53],  # I
        29: [15],  # P
        35: [3, 11, 19, 37, 55, 87],  # Group I
        42: [9],  # Fluor
        46: [35],  # Br
        88: [16],  # S
        103: [17],  # Cl
        134: [9, 17, 35, 53]  # Halogen: F,Cl,Br,I
    }

    SMARTS_SUBSTR = _smarts_substr()

    def __init__(self):
        super(MACCSFeaturizer).__init__()

    @property
    def n_features(self) -> int:
        return 167

    def _maccs(self, smiles: List[str]) -> Tuple[np.ndarray, List[Mol]]:
        maccs, mols = [], []

        for i, smile in enumerate(tqdm(smiles, desc="_maccs")):
            mol = Chem.MolFromSmiles(smile)
            mols.append(mol)

            if mol is None:
                warnings.warn(f"could not parse smile {i}: {smile}")

                maccs.append(None)
            else:
                _maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs.append(np.array(_maccs))

        return np.stack([m for m in maccs if not None]), mols

    def __call__(self, smiles: List[str]) -> np.ndarray:
        return self._maccs(smiles)[0]

    def _atomic_attribution(self, mol: Mol, feature_attribution: np.ndarray, num_atoms: Optional[int] = None) -> np.ndarray:
        """adapted/based on the implementation by J. Schimunek"""

        num_atoms = mol.GetNumAtoms() if not num_atoms else num_atoms

        idx_maccs = list(MACCSFeaturizer.SMARTS_SUBSTR.keys())
        idx_maccs_atomnumbs = list(MACCSFeaturizer.SMARTS_ATOMIC_NUMBER.keys())

        atomic_attribution = np.zeros(num_atoms)

        for maccs_idx in idx_maccs:
            # Substructure features
            pattern = MACCSFeaturizer.SMARTS_SUBSTR[maccs_idx]
            attribution_value = feature_attribution[maccs_idx]
            substructures = mol.GetSubstructMatches(pattern)

            attribution_submol = np.zeros(num_atoms)
            count_atoms = 0

            for atom_idx in range(num_atoms):
                for structure in substructures:
                    if atom_idx in structure:
                        attribution_submol[atom_idx] += attribution_value
                        count_atoms += 1
            if count_atoms != 0:
                attribution_submol = attribution_submol / count_atoms

            atomic_attribution += attribution_submol

            # Count features
            # MACCS feature: 130
            atomic_attribution = MACCSFeaturizer.maccs_count_features(maccs_idx, substructures, feature_attribution, num_atoms,
                                                                      atomic_attribution,
                                                                      feature_idx1=124, feature_idx2=130)

            # MACCS feature: 127
            atomic_attribution = MACCSFeaturizer.maccs_count_features(maccs_idx, substructures, feature_attribution, num_atoms,
                                                                      atomic_attribution,
                                                                      feature_idx1=143, feature_idx2=127)

            # MACCS feature: 138:
            atomic_attribution = MACCSFeaturizer.maccs_count_features(maccs_idx, substructures, feature_attribution, num_atoms,
                                                                      atomic_attribution,
                                                                      feature_idx1=153, feature_idx2=138)

            # MACCS features: 140, 146, 159
            ## 159
            if maccs_idx == 164 and len(substructures) > 1:
                attribution_value = feature_attribution[159]

                attribution_submol = np.zeros(num_atoms)
                count_atoms = 0

                for atom_idx in range(num_atoms):
                    for structure in substructures:
                        if atom_idx in structure:
                            attribution_submol[atom_idx] += attribution_value
                            count_atoms += 1
                if count_atoms != 0:
                    attribution_submol = attribution_submol / count_atoms

                atomic_attribution += attribution_submol

                ## 146
                if len(substructures) > 2:
                    attribution_value = feature_attribution[146]

                    attribution_submol = np.zeros(num_atoms)
                    count_atoms = 0

                    for atom_idx in range(num_atoms):
                        for structure in substructures:
                            if atom_idx in structure:
                                attribution_submol[atom_idx] += attribution_value
                                count_atoms += 1
                    if count_atoms != 0:
                        attribution_submol = attribution_submol / count_atoms

                    atomic_attribution += attribution_submol

                    ## 140
                    if len(substructures) > 3:
                        attribution_value = feature_attribution[140]

                        attribution_submol = np.zeros(num_atoms)
                        count_atoms = 0

                        for atom_idx in range(num_atoms):
                            for structure in substructures:
                                if atom_idx in structure:
                                    attribution_submol[atom_idx] += attribution_value
                                    count_atoms += 1
                        if count_atoms != 0:
                            attribution_submol = attribution_submol / count_atoms

                        atomic_attribution += attribution_submol

            # MACCS feature 142
            atomic_attribution = MACCSFeaturizer.maccs_count_features(maccs_idx, substructures, feature_attribution, num_atoms,
                                                                      atomic_attribution,
                                                                      feature_idx1=161, feature_idx2=142)

            # MACCS feature 145
            atomic_attribution = MACCSFeaturizer.maccs_count_features(maccs_idx, substructures, feature_attribution, num_atoms,
                                                                      atomic_attribution,
                                                                      feature_idx1=163, feature_idx2=145)

            # MACCS feature 149
            atomic_attribution = MACCSFeaturizer.maccs_count_features(maccs_idx, substructures, feature_attribution, num_atoms,
                                                                      atomic_attribution,
                                                                      feature_idx1=160, feature_idx2=149)

        # Atomic number features
        for idx_maccs_atomnumb in idx_maccs_atomnumbs:
            maccs_feature = MACCSFeaturizer.SMARTS_ATOMIC_NUMBER[idx_maccs_atomnumb]
            attribution_value = feature_attribution[idx_maccs_atomnumb]

            attribution_submol = np.zeros(num_atoms)
            count_atoms = 0

            for atom_idx in range(num_atoms):
                if atom_idx in maccs_feature:
                    attribution_submol[atom_idx] += attribution_value
                    count_atoms += 1
            if count_atoms != 0:
                attribution_submol = attribution_submol / count_atoms

            atomic_attribution += attribution_submol

        # MACCS 125: Aromatic rings
        atomic_attribution = MACCSFeaturizer.maccs_125_aromaticrings(mol, feature_attribution, num_atoms, atomic_attribution)

        # MACCS 166: Fragments
        atomic_attribution = MACCSFeaturizer.maccs_166_fragments(mol, feature_attribution, num_atoms, atomic_attribution)

        return atomic_attribution

    def atomic_attributions(self, smiles: List[str], feature_attributions: np.ndarray) -> List[np.ndarray]:
        assert len(smiles) == len(
            feature_attributions), f"provided number of smiles {len(smiles)} does not match number of features {len(feature_attributions)}"

        _, mols = self._maccs(smiles)

        atomic_attributions = []
        for i, (smile, mol) in enumerate(zip(smiles, mols)):
            if mol is None:
                raise ValueError(f"could not process smile/molecule {i}: {smile}")

            atomic_attribution = self._atomic_attribution(mol, feature_attributions[i])

            atomic_attributions.append(atomic_attribution)

        return atomic_attributions

    @staticmethod
    def maccs_count_features(maccs_idx: int, substructures, feature_attribution: np.ndarray, num_atoms: int, atomic_attribution: np.ndarray,
                             feature_idx1: int, feature_idx2: int) -> np.ndarray:
        """based on the implementation by J. Schimunek"""

        if maccs_idx == feature_idx1 and len(substructures) > 1:
            attribution_value = feature_attribution[feature_idx2]

            weights_submol = np.zeros(num_atoms)
            count_atoms = 0

            for atom_idx in range(num_atoms):
                for structure in substructures:
                    if atom_idx in structure:
                        weights_submol[atom_idx] += attribution_value
                        count_atoms += 1
            if count_atoms != 0:
                weights_submol = weights_submol / count_atoms

            atomic_attribution += weights_submol

        return atomic_attribution

    @staticmethod
    def isRingAromatic(mol: Mol, ringbond: Tuple[int, ...]) -> bool:
        """based on the implementation by J. Schimunek"""

        for id in ringbond:
            if not mol.GetBondWithIdx(id).GetIsAromatic():
                return False
        return True

    @staticmethod
    def maccs_125_aromaticrings(mol: Mol, feature_attribution: np.ndarray, num_atoms: int, atomic_attribution: np.ndarray) -> np.ndarray:
        """based on the implementation by J. Schimunek"""

        attribution_value = feature_attribution[125]
        substructure = list()
        ri = mol.GetRingInfo()
        ringcount = ri.NumRings()
        rings = ri.AtomRings()
        ringbonds = ri.BondRings()

        if ringcount > 1:
            for ring_idx in range(ringcount):
                ring = rings[ring_idx]
                ringbond = ringbonds[ring_idx]

                is_aromatic = MACCSFeaturizer.isRingAromatic(mol, ringbond)
                if is_aromatic == True:
                    substructure.append(ring)

            weights_submol = np.zeros(num_atoms)
            count_atoms = 0

            for atom_idx in range(num_atoms):
                for structure in substructure:
                    if atom_idx in structure:
                        weights_submol[atom_idx] += attribution_value
                        count_atoms += 1
            if count_atoms != 0:
                weights_submol = weights_submol / count_atoms

            atomic_attribution += weights_submol

        return atomic_attribution

    @staticmethod
    def maccs_166_fragments(mol: Mol, feature_attribution: np.ndarray, num_atoms: int, atomic_attribution: np.ndarray) -> np.ndarray:
        """based on the implementation by J. Schimunek"""

        attribution_value = feature_attribution[166]
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:

            weights_submol = np.zeros(num_atoms)
            count_atoms = 0

            for atom_idx in range(num_atoms):
                for structure in frags:
                    if atom_idx in structure:
                        weights_submol[atom_idx] += attribution_value
                        count_atoms += 1
            if count_atoms != 0:
                weights_submol = weights_submol / count_atoms

            atomic_attribution += weights_submol

        return atomic_attribution


def _all_patterns():
    """based/adapted on the implementation by J. Schimunek"""

    with open(Path(PurePosixPath(__file__)).parent / "resources/tox_smarts.json") as file:
        smarts_list = [s[1] for s in json.load(file)]

    # Code does not work for this case
    assert len([s for s in smarts_list if ("AND" in s) and ("OR" in s)]) == 0

    # Chem.MolFromSmarts takes a long time so it pays of to parse all the smarts first
    # and then use them for all molecules. This gives a huge speedup over existing code.
    # a list of patterns, whether to negate the match result and how to join them to obtain one boolean value

    all_patterns = []
    for smarts in smarts_list:
        patterns = []  # list of smarts-patterns
        # value for each of the patterns above. Negates the values of the above later.
        negations = []

        if " AND " in smarts:
            smarts = smarts.split(" AND ")
            merge_any = False  # If an " AND " is found all "subsmarts" have to match
        else:
            # If there is an " OR " present it"s enough is any of the "subsmarts" match.
            # This also accumulates smarts where neither " OR " nor " AND " occur
            smarts = smarts.split(" OR ")
            merge_any = True

        # for all subsmarts check if they are preceded by "NOT "
        for s in smarts:
            neg = s.startswith("NOT ")
            if neg:
                s = s[4:]
            patterns.append(Chem.MolFromSmarts(s))
            negations.append(neg)

        all_patterns.append((patterns, negations, merge_any))

    return all_patterns


class ToxFeaturizer():
    ALL_PATTERNS = _all_patterns()

    def __init__(self):
        super(ToxFeaturizer, self).__init__()

    @property
    def n_features(self) -> int:
        return 826

    def _tox(self, smile_or_mol: Union[str, Mol]) -> np.ndarray:
        """
        based/adapted on the implementation by J. Schimunek

        Matches the tox patterns against a molecule. Returns a boolean array
        """

        mol = Chem.MolFromSmiles(smile_or_mol) if isinstance(smile_or_mol, str) else smile_or_mol

        mol_features = []
        for patts, negations, merge_any in ToxFeaturizer.ALL_PATTERNS:
            matches = [mol.HasSubstructMatch(p) for p in patts]
            matches = [m != n for m, n in zip(matches, negations)]
            if merge_any:
                pres = any(matches)
            else:
                pres = all(matches)
            mol_features.append(pres)

        return np.array(mol_features)

    def _toxs(self, smiles: List[str]) -> Tuple[np.ndarray, List[Mol]]:

        toxs, mols = [], []
        for i, smile in enumerate(tqdm(smiles, desc="_toxs")):
            mol = Chem.MolFromSmiles(smile)
            mols.append(mol)

            if mol is None:
                warnings.warn(f"could not parse smile {i}: {smile}")

                toxs.append(None)
            else:
                _tox = self._tox(mol)
                toxs.append(_tox)

        return np.stack([t for t in toxs if not None]), mols

    def __call__(self, smiles: List[str]) -> np.ndarray:
        """returns a binary numpy array"""

        return self._toxs(smiles)[0]

    def _atomic_attribution(self, mol: Mol, feature_attribution: np.ndarray, num_atoms: Optional[int] = None) -> np.ndarray:
        """adapted/based on the implementation by J. Schimunek"""

        num_atoms = mol.GetNumAtoms() if not num_atoms else num_atoms

        atomic_attribution = np.zeros(num_atoms)

        tox_idx = 0

        for patts, negations, merge_any in ToxFeaturizer.ALL_PATTERNS:
            attribution_value = feature_attribution[tox_idx]
            attribution_submol = np.zeros(num_atoms)
            count_atoms = 0
            for atom_idx in range(num_atoms):
                for i in range(len(negations)):
                    neg = negations[i]
                    pattern = patts[i]

                    substructures = mol.GetSubstructMatches(pattern)
                    for structure in substructures:
                        atom_in_sub = list()

                        if str(neg) == "False":
                            if atom_idx in structure:
                                atom_in_sub.append("y")
                        elif str(neg) == "True":
                            if atom_idx not in structure:
                                atom_in_sub.append("y")

                        if "y" in str(atom_in_sub):
                            attribution_submol[atom_idx] += attribution_value
                            count_atoms += 1

            if count_atoms != 0:
                attribution_submol = attribution_submol / count_atoms
                atomic_attribution += attribution_submol

            tox_idx += 1

        return atomic_attribution

    def atomic_attributions(self, smiles: List[str], feature_attributions: np.ndarray) -> List[np.ndarray]:
        assert len(smiles) == len(
            feature_attributions), f"provided number of smiles {len(smiles)} does not match number of features {len(feature_attributions)}"

        _, mols = self._toxs(smiles)

        atomic_attributions = []
        for i, (smile, mol) in enumerate(zip(smiles, mols)):
            if mol is None:
                raise ValueError(f"could not process smile/molecule {i}: {smile}")

            atomic_attribution = self._atomic_attribution(mol, feature_attributions[i])

            atomic_attributions.append(atomic_attribution)

        return atomic_attributions
