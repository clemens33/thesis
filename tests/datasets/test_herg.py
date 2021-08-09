import pytest


class TestHERGClassifierDataset():

    @pytest.mark.parametrize("use_cache, featurizer_name, featurizer_kwargs",
                             [
                                 (False, "combined", {"radius": 3,
                                                      "fold": 1024,
                                                      "return_count": False}),
                                 (True, "combined", {"radius": 3,
                                                     "fold": None,
                                                     "return_count": False}),

                             ])
    def test_prepare_data(self, use_cache, featurizer_name, featurizer_kwargs):
        from datasets.herg import HERGClassifierDataset

        dm = HERGClassifierDataset(
            batch_size=128,
            use_cache=use_cache,
            featurizer_name=featurizer_name,
            featurizer_kwargs=featurizer_kwargs
        )

        dm.prepare_data()

        # TODO add asserts
        assert True
