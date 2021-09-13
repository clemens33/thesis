import numpy as np
import pytest


@pytest.fixture(scope="session", params=[
    (
            ['O=C2CN(C(=O)c1cc(F)cc(F)c1)CCN2c5ccc(OC4CCN(C3CCC3)CC4)cc5',
             'c1cc(C)nc(c12)cccc2N(CC3)CCN3CCCc(ccc4)c(c45)OCC(=O)N5C',
             'Cn1c(CCCCN2CC3[C@@](C2)(C3)c2ccc(cc2)C(F)(F)F)nnc1c1cscc1',
             'Fc1cccc2c1c(=O)n(c(n2)C)c1ccc(cc1)OCCCN1CCCC1',
             'O=C(N(C)C)NC1CCC(CC1)CCN1[C@@H]2CC[C@H]1C[C@H](C2)Oc1cccc(c1)C(=O)N',
             'O=C(N)N1c2c(C=Cc3c1cccc3)cccc2',
             'Fc1cnc2c(c1)n(CCN1CCC(CC1)c1[nH]c3c(n1)cc(c(c3)C)Cl)c(=O)cc2',
             'ONC(=O)C=Cc1ccc2c(c1)CN(C2)S(=O)(=O)c1ccc(cc1)C(F)(F)F',
             'Nc8ccc(c7nnc(C6CCN(Cc5ccc(c3nc2nc(N1CCN(CCO)CC1)ncc2cc3c4ccccc4)cc5)CC6)[nH]7)cn8',
             'CC(C)CN1CC2CN(CC(C)C)CC(C1)C23CCCCC3',
             'O=C(N1CCC(CC1)CNc1ncccn1)OCc1ccc(cc1)C(C)(C)C',
             'Cc1nn(c(c1c1nnc(n1C)CCCCN1CC2[C@@](C1)(C2)c1ccc(cc1)C(F)(F)F)C)C',
             'COc1cc2c(cc1OC)CCN([C@@H]2Cc1ccc(cc1)Oc1cc(ccc1O)C[C@H]1N(C)CCc2c1cc(OC)c(c2)OC)C',
             'O=C1N(CCN1c1ccc(cc1)C(C)(C)C)CCN1Cc2c(C1)cccc2',
             'COc1ccc2c(n1)c(CCC13CCC(CC1)(CO3)NCc1ccc3c(n1)NC(=O)CO3)c(cn2)CO',
             'O=C(NCCN1CCCCC1)c5ccc(CN4CCC(NC(=O)c3cc(=O)c2ccc(F)cc2o3)CC4)cc5',
             'CN[C@H]1CCN(C1)C(=O)c1ccc(cc1)Nc1ncc(c(n1)c1cnc(n1C(C)C)C)F',
             'O=C1COc2c(N1)nc(cc2)CNC12CCC(CC1)(OC2)CCc1c(F)cnc2c1nc(cc2)OC1CCC1',
             'COCCCn1cc(c2c1cccc2F)CN(C(=O)C1CNCCC1(O)c1ccc(c(c1)F)F)C1CC1',
             'C1CC[C@@H](C)N1CCc(cc2)cc(c23)ccc(n3)-c(cn4)c(C)nc4N5CCCC5',
             'CN1C2CCCC1CC(C2)NC(=O)c1cccc2c1nc(o2)N1C(C)CNCC1C',
             'Fc1cc(ccc1C1CC2CCC(C1)N2CCCSc1nnc(n1C)c1ocnc1C)C(F)(F)F',
             'CNC(=O)c1cc(ccc1OC(F)F)c1ccc2c(n1)nc(nc2N1CCOC[C@H]1C)N1CCOC[C@H]1C',
             'Clc1ccc2c(n1)N1[C@H](C)CNC[C@H]1C2',
             'COc1cnc2c(c1)n(C[C@H](C1CCC(CC1)NCc1ccc3c(n1)NC(=O)CO3)N)c(=O)cc2',
             'CN1CCN(CC1)CCCNc1nnc(o1)c1ccc(cc1)NC(=O)c1ccccc1F',
             'O=C1OC2(CN1c1ccccc1)CCC(CC2)c1nc2c([nH]1)cc(cc2)C(F)(F)F',
             'CCCCCCCN(CC#CCOC(c1ccccc1)c1ccccc1)CC',
             'CC(Cn1cncc1)NC(=O)NC1CCN(CC1)Cc1ccn(c1)c1ccc(cc1)C(F)(F)F',
             'Fc1cc2c([nH]c(-c3ccccc3)c2[C@@H]2C[NH2+]CC[C@@H]2F)cc1',
             'O1[C@H](CC)[C@](O)(C)[C@H](O)[C@@H](C)C(=O)[C@@H](C[C@](O)(C)[C@H](O[C@@H]2O[C@@H](C[C@H](N(C)C)[C@H]2O)C)[C@@H](C)[C@H](O[C@@H]2O[C@@H](C)[C@H](O)[C@](O)(C2)C)[C@@H](C)C1=O)C',
             'Cn5c(=O)ccc4ccc(CN3CCC(NC(=O)c2cc(=O)c1ccc(F)cc1o2)CC3)cc45',
             'Cn1c(SCCCN2C[C@@H]3[C@](C2)(C3)c2ccc(cc2)OC(F)(F)F)nnc1c1ocnc1C',
             'CCc1ccccc1C(=O)N(C1CNCC1)CC1CCC1',
             'OCCNCCN1CCCc2c1ccc(c2)NC(=N)c1cccs1',
             'CCCNS(=O)(=O)c1ccc(cc1)c1ccc(cc1)CCN1CCCC1C',
             'O=C1NCc2c(N1c1c(Cl)cccc1Cl)nc(nc2c1ccccc1Cl)N1CCc2c(C1)cn[nH]2',
             'C1COCCC1Cn2cc(c(c23)cccc3Cl)-c4sc(CC)c(n4)C[C@@H](C)NCCO',
             'O=C1COc2c(N1)nc(cc2)CNC12CCC(CC1)(OC2)CC1(O)Cn2c3c1c(F)c(Cl)nc3ccc2=O',
             'O(c1cc2c([nH]c(C)c2C)cc1)c1ncnc2c1cc(OC)c(OCCN1CCN(CC1)CC(=O)C)c2',
             'CNS(=O)(=O)Cc2ccc1[nH]cc(CCN(C)C)c1c2',
             'O=C(N1CCC(CC1)N1CCC(CC1)Oc1ccc(c(c1)Cl)Cl)NS(=O)(=O)c1ccccc1C',
             'c1cc(C)nc(c12)cccc2N(CC3)CCN3CCc(cc4)cc(c45)OCC(=O)N5',
             'CCN(CCNC(=O)c1ccc(cc1)NC(=O)C)CC',
             'Fc1ccc2c(c1)onc2C1CCN(CC1)CCc1c(C)nc2n(c1=O)CCCC2O',
             'c1ccccc1Cn(cc2)c(c23)ncnc3OC4CCN(CC4)Cc(cc5)ncc5C',
             'Cn1c(CCCCN2CC3[C@@](C2)(C3)c2ccc(cc2)C(F)(F)F)nnc1c1ccnnc1',
             'COCCOCC#Cc5cc(C1N=NC3=C1Cc4cc(CN2CCN(C)CC2)ccc34)cs5',
             'CCCc1nn(c2c1nc([nH]c2=O)c1cc(ccc1OCC)S(=O)(=O)N1CCN(CC1)C)C',
             'COc1cc(ccc1CC(=O)Nc1cn(nc1C)C)Oc1ccnc2c1ccc(c2)OC',
             'Brc1ccc(cc1)c1c(Cn2ncnc2)c(nn1c1ccccc1Cl)c1nnc(s1)C(C)(C)C',
             'c1cccc(c12)[nH]c(-c3ccccc3)c2[C@@H](C[C@H]45)C[C@H](CC5)N4CCc6ccccc6',
             'C1CN(CCN1CCOCC(=O)O)C(C2=CC=CC=C2)C3=CC=C(C=C3)Cl',
             'c1ccnc(F)c1-c(c2)ccc(c2[C@]34N=C(N)OC4)Oc5c3cc(cn5)-c(cc6)ccc6C',
             'N#Cc1cnc(cn1)Nc1ncc(c(c1)NC[C@H]1CNCCO1)c1ncc(s1)C',
             'CN1CC2[C@@](C1)(CCCC2)c1ccc(c(c1)Cl)Cl',
             'Cc1ccc(cc1)S(=O)(=O)NC(=O)N1CCC(CC1)N1CCC(CC1)Oc1ccc(cc1C)Cl',
             'N[C@H](C1CCN(CC1)C(=O)c1cnc2n(c1C)nc(c2)C)Cc1cc(F)c(cc1F)F',
             'CC(NC(=O)c1cn(c2cccc(c2)c2ccc(cc2)S(=O)(=O)C)c2c(c1=O)cccn2)C',
             'O=C([C@H]1CNCC[C@]21OCc1c2cc(F)c(c1)F)N(C1CC1)Cc1ccccc1Cl',
             'O=C(N1CCC(C1)NC1CCC(CC1)(O)c1ccc(cn1)c1ncccn1)CNC(=O)c1cccc(c1)C(F)(F)F',
             'O=C(NS(=O)(=O)c1ccc(cc1)OCCCN1CCCCC1)Nc1ccccc1C(F)(F)F',
             'O=C1CC2C(C1)CN(C2)c1ccc(cc1F)N1C[C@@H](OC1=O)Cn1nncc1',
             'N#Cc1ccc(cc1Cl)N([C@H]1CCN(C1)S(=O)(=O)C)Cc1ccccc1C',
             'CCCCCCCCCCCCc([n+]1C)cccc1CCCCCCCCCCCC',
             'O=C(NC1CN(C1)C1CCC(CC1)c1ccccc1O)CNc1ncnc2c1cc(cc2)C(F)(F)F',
             'O=C(c1ccc(nc1)N1CCC2(CC1)CCN(C2)S(=O)(=O)C)Nc1cc(ccc1N)c1cccs1',
             'CN1CC2CC1CN2c1ccc(cn1)c1ccc2c(c1)cco2',
             'COc1ccc(cn1)c1cc(cnc1N)c1ccc(cc1)S(=O)(=O)N1CCN(CC1)C',
             'O=c1oc2cc(OCCCCN3CCC(CC3)c3noc4c3ccc(c4)F)ccc2c(c1)c1ccccc1',
             'CCN(C(=O)Cc1ccc(S(C)(=O)=O)cc1)C4CCN(CC[C@H](c2ccc(S(C)(=O)=O)cc2)c3cc(F)cc(F)c3)CC4',
             'N#Cc1ccc2c(c1)N(CC(CN(C)C)C)c1c(S2=O)cccc1',
             'COc(c1)ccc(c12)n(C)c(=O)cc2NC3CCN(CC3)Cc(c4)ccc(c45)OCO5',
             '[H]N(C(=O)c2ccc(c1ccc(Cl)cc1)o2)c3cccc(C)c3',
             'c1cccc(F)c1S(=O)(=O)c(cn2)ccc2/C=C/c(c(c3)C#N)ccc3F',
             'N#Cc1ccc(cc1Cl)N([C@H]1CCN(C1)C)Cc1ccccc1C(F)(F)F',
             'COc1ccc(cc1)C1CCC(CC1)N1CC(C1)NC(=O)CNC(=O)c1cccc(c1)C(F)(F)F',
             'COc1ccc(c2c1cccc2)S(=O)(=O)N1CC(c2c1cccc2)C(=O)N1CCNCCC1',
             'O=C(c1ccc(nc1)N1CCC2(CC1)CNCC2)Nc1cc(ccc1N)c1n[nH]cc1',
             'CCc6nc5cc4CCN(C(C)CCSc3nnc(c1cccc2nc(C)ccc12)n3C)CCc4c(C)c5o6',
             'Nc1nc2cc(sc2c(n1)NC1CCNCC1)c1ccc(cc1)C(F)(F)F',
             'Fc1ccc(c(c1)F)c1nnc(n1C)CCCCCN1CC2[C@@](C1)(C2)c1ccc(cc1)C(F)(F)F',
             'CCCCN1CCCCC1C(=O)Nc2c(C)cccc2C',
             'O=C(c1ccccc1F)Nc1ccc(cc1)c1nnc(o1)NCCCCN1CCOCC1',
             'Cc1ccc2c(n1)cccc2c1nnc(n1C)SCCCN1CCc2c(CC1)cc1c(c2)oc(n1)N(C)C',
             'CN(Cc1ccc2c(c1)CC[C@@H](C2)N(C(=O)c1ccc(cc1)c1ccc(cn1)Cl)C)C',
             'CC[C@@H]3OC(=O)[C@H](C)[C@@H](OC1C[C@@](C)(OC)[C@@H](O)[C@H](C)O1)[C@H](C)[C@@H](OC2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(OC)C[C@@H](C)C(=NOCOCCOC)[C@@H](C)[C@@H](O)[C@]3(C)O',
             'Cc6ccc5c(c4nnc(SCCC(C)N3CCc2cc1CN(C(C)C)C(=O)c1cc2CC3)n4C)cccc5n6',
             'C1CC[C@@H](C)N1CCc(cc2)cc(c23)ccc(n3)-c4c(C)c(no4)-c(cc5)ccc5Cl',
             'CCN(C(=O)Cc1ccc(cc1)S(=O)(=O)C)C1CCN(CC1)Cc1ccc(cc1)C(F)(F)F',
             'C[C@@H]1CN(CCN1c1ncc(cn1)OCc1ccc(cc1)CS(=O)(=O)C)c1nnc(o1)C(F)(F)F',
             'N#Cc1ccc(cc1)Cn1cncc1CNC1CCN(C1)C(=O)c1ccc[nH]c1=O',
             'CN1CC2CC1CN2c1ncc(cn1)c1ccc2c(c1)c[nH]n2',
             'CO/C=C(\\[C@H]1CC2N(C[C@@H]1CC)CCC12C(=O)Nc2c1cccc2)/C(=O)OC',
             'O=c6[nH]nc(C2CC3(c1ccc(F)cc1)NC2CCC3NCc4cc(OC(F)(F)F)ccc4OC5CC5)[nH]6',
             'c1c(Cl)c(N)cc(OC)c1C(=O)N[C@H](CC2)[C@H](OC)C[C@@H]2CCCOc3ccc(F)cc3',
             'COc1ccc(cc1)CC(=O)Nc1cc2c(cc1[N+](=O)[O-])OC(C(C2NC1CC1)O)(C)C',
             'COc1cccc2c1OC(c1ccc(cc1)OCCCN1CCCCC1)C(S2(=O)=O)C',
             'CN(CCc1sc2c(c1C(c1cccnc1)C)cccc2)C',
             'COc1c(C)cc(nc1C)C1(N=C(c2c1cccc2F)N)c1cccc(c1)c1cncnc1']
    )
])
def herg_sample_smiles(request):
    smiles = request.param

    return smiles


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

    @pytest.mark.parametrize("smile, n_jobs, radius, fold, use_chirality, use_features, return_count",
                             [
                                 ("c1cc(C)nc(c12)cccc2N(C[C@@H]3C)CCN3CCc(ccc4)c(c45)ccc(=O)n5C", 0, 3, None, False, False, True),
                                 ("FC(C(COc1ccccc1CN1CCC2(CC1)CCN(CC2)C(=O)c1ccncc1)(F)F)F", 4, 3, 128, False, False, False),
                             ])
    def test_atomic_mapping(self, smile, n_jobs, radius, fold, use_chirality, use_features, return_count):
        """tests if direct atomic attribution calculation matches calculation from atomic mapping"""

        from datasets.featurizer import ECFPFeaturizer, _atomic_attribution_from_mapping

        featurizer = ECFPFeaturizer(n_jobs=n_jobs, radius=radius, fold=fold, use_features=use_features, use_chirality=use_chirality,
                                    return_count=return_count)
        features = featurizer([smile])
        atomic_mapping = featurizer._atomic_mapping(smile)

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution_reference = featurizer.atomic_attributions([smile], dummy_attribution)
        atomic_attribution = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution[0])

        assert np.allclose(atomic_attribution, atomic_attribution_reference[0])

        # run again and again with same mapping but different random attributions
        dummy_attribution2 = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution_reference2 = featurizer.atomic_attributions([smile], dummy_attribution2)
        atomic_attribution2 = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution2[0])

        assert np.allclose(atomic_attribution2, atomic_attribution_reference2[0])

        # run again and again with same mapping but different random attributions
        dummy_attribution3 = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution_reference3 = featurizer.atomic_attributions([smile], dummy_attribution3)
        atomic_attribution3 = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution3[0])

        assert np.allclose(atomic_attribution3, atomic_attribution_reference3[0])

    @pytest.mark.parametrize("n_jobs, radius, fold, use_chirality, use_features, return_count",
                             [
                                 (0, 3, None, False, False, True),
                                 (4, 3, 128, False, False, False),
                             ])
    def test_atomic_mappings(self, herg_sample_smiles, n_jobs, radius, fold, use_chirality, use_features, return_count):
        """tests if direct atomic attribution calculation matches calculation from atomic mapping"""

        from datasets.featurizer import ECFPFeaturizer, atomic_attributions_from_mappings

        smiles = herg_sample_smiles

        featurizer = ECFPFeaturizer(n_jobs=n_jobs, radius=radius, fold=fold, use_features=use_features, use_chirality=use_chirality,
                                    return_count=return_count)
        features = featurizer(smiles)
        atomic_mappings = featurizer.atomic_mappings(smiles)

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attributions_reference = featurizer.atomic_attributions(smiles, dummy_attribution)
        atomic_attributions = atomic_attributions_from_mappings(atomic_mappings, dummy_attribution, n_jobs=n_jobs)

        for aa, aar in zip(atomic_attributions, atomic_attributions_reference):
            assert np.allclose(aa, aar)

        # run again with different dummy attribution but with same atomic mappings
        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attributions_reference = featurizer.atomic_attributions(smiles, dummy_attribution)
        atomic_attributions = atomic_attributions_from_mappings(atomic_mappings, dummy_attribution, n_jobs=n_jobs)

        for aa, aar in zip(atomic_attributions, atomic_attributions_reference):
            assert np.allclose(aa, aar)


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

    @pytest.mark.parametrize("smile, n_jobs",
                             [
                                 ("c1cc(C)nc(c12)cccc2N(C[C@@H]3C)CCN3CCc(ccc4)c(c45)ccc(=O)n5C", 0),
                                 ("FC(C(COc1ccccc1CN1CCC2(CC1)CCN(CC2)C(=O)c1ccncc1)(F)F)F", 4),
                             ])
    def test_atomic_mapping(self, smile, n_jobs):
        """tests if direct atomic attribution calculation matches calculation from atomic mapping"""

        from datasets.featurizer import MACCSFeaturizer, _atomic_attribution_from_mapping

        featurizer = MACCSFeaturizer(n_jobs) if n_jobs else MACCSFeaturizer()
        atomic_mapping = featurizer._atomic_mapping(smile)

        features = featurizer([smile])

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution[0])
        atomic_attribution_reference = featurizer.atomic_attributions([smile], dummy_attribution)

        assert np.allclose(atomic_attribution, atomic_attribution_reference[0])

        # run again and again with same mapping but different random attributions
        dummy_attribution2 = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution_reference2 = featurizer.atomic_attributions([smile], dummy_attribution2)
        atomic_attribution2 = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution2[0])

        assert np.allclose(atomic_attribution2, atomic_attribution_reference2[0])

        # run again and again with same mapping but different random attributions
        dummy_attribution3 = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution_reference3 = featurizer.atomic_attributions([smile], dummy_attribution3)
        atomic_attribution3 = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution3[0])

        assert np.allclose(atomic_attribution3, atomic_attribution_reference3[0])

    @pytest.mark.parametrize("n_jobs",
                             [
                                 (0),
                                 (4),
                             ])
    def test_atomic_mappings(self, herg_sample_smiles, n_jobs):
        """tests if direct atomic attribution calculation matches calculation from atomic mapping"""

        from datasets.featurizer import MACCSFeaturizer, atomic_attributions_from_mappings

        smiles = herg_sample_smiles

        featurizer = MACCSFeaturizer(n_jobs) if n_jobs else MACCSFeaturizer()
        atomic_mappings = featurizer.atomic_mappings(smiles)

        features = featurizer(smiles)

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attributions_reference = featurizer.atomic_attributions(smiles, dummy_attribution)
        atomic_attributions = atomic_attributions_from_mappings(atomic_mappings, dummy_attribution, n_jobs=n_jobs)

        for aa, aar in zip(atomic_attributions, atomic_attributions_reference):
            assert np.allclose(aa, aar)

        # run again with different dummy attribution but with same atomic mappings
        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attributions_reference = featurizer.atomic_attributions(smiles, dummy_attribution)
        atomic_attributions = atomic_attributions_from_mappings(atomic_mappings, dummy_attribution, n_jobs=n_jobs)

        for aa, aar in zip(atomic_attributions, atomic_attributions_reference):
            assert np.allclose(aa, aar)


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

    @pytest.mark.parametrize("smile, n_jobs",
                             [
                                 ("c1cc(C)nc(c12)cccc2N(C[C@@H]3C)CCN3CCc(ccc4)c(c45)ccc(=O)n5C", 0),
                                 ("FC(C(COc1ccccc1CN1CCC2(CC1)CCN(CC2)C(=O)c1ccncc1)(F)F)F", 4),
                             ])
    def test_atomic_mapping(self, smile, n_jobs):
        """tests if direct atomic attribution calculation matches calculation from atomic mapping"""

        from datasets.featurizer import ToxFeaturizer, _atomic_attribution_from_mapping

        featurizer = ToxFeaturizer(n_jobs) if n_jobs else ToxFeaturizer()
        atomic_mapping = featurizer._atomic_mapping(smile)

        features = featurizer([smile])

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution[0])
        atomic_attribution_reference = featurizer.atomic_attributions([smile], dummy_attribution)

        assert np.allclose(atomic_attribution, atomic_attribution_reference[0])

        # run again and again with same mapping but different random attributions
        dummy_attribution2 = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution_reference2 = featurizer.atomic_attributions([smile], dummy_attribution2)
        atomic_attribution2 = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution2[0])

        assert np.allclose(atomic_attribution2, atomic_attribution_reference2[0])

        # run again and again with same mapping but different random attributions
        dummy_attribution3 = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attribution_reference3 = featurizer.atomic_attributions([smile], dummy_attribution3)
        atomic_attribution3 = _atomic_attribution_from_mapping(atomic_mapping, dummy_attribution3[0])

        assert np.allclose(atomic_attribution3, atomic_attribution_reference3[0])

    @pytest.mark.parametrize("n_jobs",
                             [
                                 (0),
                                 (4),
                             ])
    def test_atomic_mappings(self, herg_sample_smiles, n_jobs):
        """tests if direct atomic attribution calculation matches calculation from atomic mapping"""

        from datasets.featurizer import ToxFeaturizer, atomic_attributions_from_mappings

        smiles = herg_sample_smiles

        featurizer = ToxFeaturizer(n_jobs) if n_jobs else ToxFeaturizer()
        atomic_mappings = featurizer.atomic_mappings(smiles)

        features = featurizer(smiles)

        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attributions_reference = featurizer.atomic_attributions(smiles, dummy_attribution)
        atomic_attributions = atomic_attributions_from_mappings(atomic_mappings, dummy_attribution, n_jobs=n_jobs)

        for aa, aar in zip(atomic_attributions, atomic_attributions_reference):
            assert np.allclose(aa, aar)

        # run again with different dummy attribution but with same atomic mappings
        dummy_attribution = np.random.binomial(1, 0.3, size=features.shape) * np.random.random(features.shape)
        atomic_attributions_reference = featurizer.atomic_attributions(smiles, dummy_attribution)
        atomic_attributions = atomic_attributions_from_mappings(atomic_mappings, dummy_attribution, n_jobs=n_jobs)

        for aa, aar in zip(atomic_attributions, atomic_attributions_reference):
            assert np.allclose(aa, aar)


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
