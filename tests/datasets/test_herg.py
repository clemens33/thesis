import pytest
from tempfile import mkdtemp


class TestHERGClassifierDataset():

    @pytest.fixture(scope="session")
    def cache_dir(self):
        #cache_dir = str(mkdtemp()) + "/"

        from pathlib import Path,PurePosixPath
        cache_dir = str(Path.home()) + "/.cache/herg/"
        Path(PurePosixPath(cache_dir)).mkdir(parents=True, exist_ok=True)

        return cache_dir

    @pytest.mark.parametrize("use_cache, featurizer_name, featurizer_kwargs",
                             [
                                 (False, "combined", {"radius": 3,
                                                      "fold": 1024,
                                                      "return_count": True}),

                                 (True, "combined", {"radius": 3,
                                                     "fold": None,
                                                     "return_count": False}),

                             ])
    def test_prepare_data(self, cache_dir, use_cache, featurizer_name, featurizer_kwargs):
        """basic prepare data test case"""

        from datasets.herg import HERGClassifierDataModule
        from pathlib import Path, PurePosixPath

        dm = HERGClassifierDataModule(
            batch_size=128,
            cache_dir=cache_dir,
            use_cache=use_cache,
            featurizer_name=featurizer_name,
            featurizer_kwargs=featurizer_kwargs
        )

        dm.prepare_data()

        if use_cache:
            assert len(list(Path(PurePosixPath(cache_dir)).iterdir())) > 0

    @pytest.mark.parametrize("use_labels, split_size, standardize",
                             [
                                 (["active_g100"], (0.9, 0.1, 0.0), True),
                                 (["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"], (0.8, 0.1, 0.1), True),
                                 (["active_g10"], (0.8, 0.1, 0.1), True),
                                 (["active_g10"], (0.8, 0.2), True),
                                 (["active_g10", "active_g100"], (0.9, 0.1, 0.0), True),
                                 (None, (0.8, 0.1, 0.1), True),
                                 (None, (0.8, 0.1, 0.1), False),

                             ])
    def test_setup(self, cache_dir, use_labels, split_size, standardize):
        """basic setup test case"""

        from datasets.herg import HERGClassifierDataModule

        dm = HERGClassifierDataModule(
            batch_size=128,
            cache_dir=cache_dir,
            use_cache=True,
            use_labels=use_labels,
            featurizer_name="combined",
            featurizer_kwargs={"radius": 3, "fold": None, "return_count": False},
            split_size=split_size,
            standardize=standardize
        )

        dm.prepare_data()
        dm.setup()

        assert len(dm.train_dataloader()) > 0
        assert len(dm.val_dataloader()) > 0

        if split_size[-1] == 0 or len(split_size) == 2:
            assert dm.test_dataloader() is None
        else:
            assert len(dm.test_dataloader()) > 0




