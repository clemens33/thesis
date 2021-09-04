import numpy as np
import pytest


@pytest.fixture(scope="session", params=[
    (
            [
                "FC(C(COc1ccccc1CN1CCC2(CC1)CCN(CC2)C(=O)c1ccncc1)(F)F)F",
                "CNCc1ccc(cc1Oc1ccc(cc1)Cl)C(F)(F)F",
                "N#Cc1ccc2c(c1)[nH]c(n2)C1CCC2(CC1)OC(=O)N(C2)c1ccccc1",
                "c1cc(C)cc(c12)CC(CCN(C)C)=C2[C@@H](C)c3nccnc3",
                "C1CC[C@@H](C)N1CCc(cc2)cc(c23)ccc(n3)-c4c(C)nc(s4)C",
                "C1CC[C@@H](C)N1CCc(cc2)cc(c23)ccc(n3)-c4c(C)nc(s4)N5CCOCC5",
                "c1cc(C)nc(c12)cccc2N(C[C@@H]3C)CCN3CCc(ccc4)c(c45)ccc(=O)n5C",
                "c1cc(C)nc(c12)cccc2N(CC3)CCN3CCCc(ccc4)c(c45)OCC(=O)N5C",
            ] * 2
    )
])
def sample_smiles(request):
    smiles = request.param

    return smiles


class TestECFP_Featurizer():

    @pytest.mark.parametrize("radius, fold, use_chirality, use_features, return_count, n_jobs",
                             [
                                 (1, 32, False, False, True, 2),
                                 (3, None, True, True, True, 4),
                                 (10, None, True, True, False, 2),
                                 (3, 1024, False, False, True, 3),
                                 (1, 1024, False, True, True, 0),
                             ])
    def test_fit_transform(self, sample_smiles, radius, fold, use_chirality, use_features, return_count, n_jobs):
        """basic ecfp featurizer tests"""

        from datasets.featurizer import ECFPFeaturizer

        featurizer = ECFPFeaturizer(radius=radius, fold=fold, use_features=use_features, use_chirality=use_chirality,
                                    return_count=return_count, n_jobs=n_jobs)
        desc = featurizer(sample_smiles)

        assert len(sample_smiles) == len(desc)

        if fold:
            assert desc.shape[-1] == fold

        if return_count:
            # if we return count we expect some entry to be greater than 1
            assert np.any((desc > 1))
        else:
            # if we dont return count (bit vector) all must be smaller-equal 1
            assert np.all((desc <= 1))

    @pytest.mark.parametrize("radius, fold, use_chirality, use_features",
                             [
                                 (1, 32, False, False),
                                 (3, 32, True, True),
                                 (10, 8196, True, True),
                                 (3, 1024, False, False),
                                 (1, 1024, False, False),
                             ])
    def test_fit_transform_bit_vector(self, sample_smiles, radius, fold, use_chirality, use_features):
        """test if custom bit vector implementation matches the direct as bit vector implementation."""

        from datasets.featurizer import ECFPFeaturizer
        from rdkit import Chem
        from rdkit.Chem import AllChem

        desc_expected = []
        for smile in sample_smiles:
            mol = Chem.MolFromSmiles(smile)

            fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fold, useChirality=use_chirality,
                                                        useFeatures=use_features)
            desc_expected.append(np.array(fps))

        desc_expected = np.stack(desc_expected)

        featurizer = ECFPFeaturizer(radius=radius, fold=fold, use_features=use_features, use_chirality=use_chirality, return_count=False)
        desc = featurizer(sample_smiles)

        assert np.allclose(desc, desc_expected)

    @pytest.mark.parametrize("radius, fold, use_chirality, use_features, return_count",
                             [
                                 (3, None, False, False, True),
                                 (3, 128, False, False, False),
                                 (3, 32, True, True, False),
                                 (10, 8196, True, True, False),
                                 (3, 1024, False, False, False),
                                 (1, 1024, False, False, False),
                             ])
    def test_atomic_attribution(self, sample_smiles, radius, fold, use_chirality, use_features, return_count):
        from datasets.featurizer import ECFPFeaturizer
        import numpy as np

        featurizer = ECFPFeaturizer(radius=radius, fold=fold, use_features=use_features, use_chirality=use_chirality,
                                    return_count=return_count)
        features = featurizer(sample_smiles)

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)

        atomic_attribution = featurizer.atomic_attributions(sample_smiles, dummy_attribution)

        assert len(sample_smiles) == len(atomic_attribution)


class TestMACCSFeaturizer():

    @pytest.mark.parametrize("n_jobs",
                             [
                                 (1), (4), (None),
                             ])
    def test_featurizer(self, sample_smiles, n_jobs):
        """basic maccs featurizer tests"""

        from datasets.featurizer import MACCSFeaturizer

        featurizer = MACCSFeaturizer(n_jobs) if n_jobs else MACCSFeaturizer()
        desc = featurizer(sample_smiles)

        assert len(sample_smiles) == len(desc)

    @pytest.mark.parametrize("n_jobs",
                             [
                                 (1), (4), (None),
                             ])
    def test_atomic_attribution(self, sample_smiles, n_jobs):
        """basic maccs featurizer tests"""

        from datasets.featurizer import MACCSFeaturizer

        featurizer = MACCSFeaturizer(n_jobs) if n_jobs else MACCSFeaturizer()
        features = featurizer(sample_smiles)

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)

        atomic_attribution = featurizer.atomic_attributions(sample_smiles, dummy_attribution)

        assert len(sample_smiles) == len(atomic_attribution)


class TestToxFeaturizer:

    @pytest.mark.parametrize("n_jobs",
                             [
                                 (1), (4), (None),
                             ])
    def test_featurizer(self, sample_smiles, n_jobs):
        "basic featurizer tests"
        from datasets.featurizer import ToxFeaturizer

        featurizer = ToxFeaturizer(n_jobs) if n_jobs else ToxFeaturizer()
        features = featurizer(sample_smiles)

        assert len(sample_smiles) == len(features)

    @pytest.mark.parametrize("n_jobs",
                             [
                                 (4),
                                 (1),
                                 (None),
                             ])
    def test_atomic_attribution(self, sample_smiles, n_jobs):
        """basic maccs attribution tests"""

        from datasets.featurizer import ToxFeaturizer

        featurizer = ToxFeaturizer(n_jobs) if n_jobs else ToxFeaturizer()
        features = featurizer(sample_smiles)

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)

        atomic_attribution = featurizer.atomic_attributions(sample_smiles, dummy_attribution)

        assert len(sample_smiles) == len(atomic_attribution)


@pytest.mark.parametrize("smiles, reference_smiles, provide_labels, provide_preds, indefinite_labels, seed",
                         [
                             (["CNCc1ccc(cc1Oc1ccc(cc1)Cl)C(F)(F)F"], [("n1ccccc1", 1)], True, True, 0, 1),
                             (None, [("n1ccccc1", 0), ("c1cc(C)ccc1C", 1)], True, True, 3, 1),
                             (None, ["n1ccccc1", "n1ccccc1C", "c1cc(C)ccc1C"], True, True, 0, 1),
                             (None, ["n1ccccc1", "n1ccccc1C", "c1cc(C)ccc1C"], False, True, 3, 1),
                             (None, ["n1ccccc1", "n1ccccc1C", "c1cc(C)ccc1C"], True, False, 3, 1),
                             (None, ["n1ccccc1", "n1ccccc1C", "c1cc(C)ccc1C"], False, False, 3, 1),
                             (None, ["n1ccccc1", "c1cc(C)ccc1C"], False, False, 3, 1),
                         ])
def test_match(sample_smiles, smiles, reference_smiles, provide_labels, provide_preds, indefinite_labels, seed):
    from datasets.featurizer import calculate_ranking_scores
    from rdkit import Chem

    rng = np.random.default_rng(seed)

    smiles = sample_smiles if smiles is None else smiles

    dummy_attribution = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)

        dummy_attribution.append(rng.random(mol.GetNumAtoms()))

    # generate dummy labels, preds
    labels = rng.integers(0, 2, size=len(smiles)).astype(float) if provide_labels else None
    preds = rng.integers(0, 2, size=len(smiles)).astype(float) if provide_preds else None

    if indefinite_labels > 0 and provide_labels:
        nan_indices = rng.integers(0, len(smiles), indefinite_labels)
        labels[nan_indices] = float("nan")

    result, reference_results, df = calculate_ranking_scores(smiles, reference_smiles, dummy_attribution, labels=labels, preds=preds)

    assert len(df) == len(smiles)
    assert len(reference_results) == len(reference_smiles)
