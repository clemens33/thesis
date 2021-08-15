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
        from datasets.herg import HERGClassifierDataModule
        from tempfile import mkdtemp
        from pathlib import Path, PurePosixPath

        cache_dir = mkdtemp()

        dm = HERGClassifierDataModule(
            batch_size=128,
            cache_dir=str(cache_dir) + "/",
            use_cache=use_cache,
            featurizer_name=featurizer_name,
            featurizer_kwargs=featurizer_kwargs
        )

        dm.prepare_data()

        if use_cache:
            assert len(list(Path(PurePosixPath(cache_dir)).iterdir())) > 0
