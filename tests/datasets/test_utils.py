import pytest


@pytest.mark.parametrize("nr_samples, split_size", [
    (100, (5, 0, 1)),
    (7885, (5, 0)),
    (354, (7, 3, 4)),
])
def test_split_kfold(nr_samples, split_size):
    """basic tests"""
    from datasets.utils import split_kfold
    import numpy as np

    expected_indices = np.arange(nr_samples)

    train_indices, val_indices, test_indices = split_kfold(nr_samples, split_size)

    returned_indices = [train_indices, val_indices]
    returned_indices += [test_indices] if test_indices is not None else []
    returned_indices = np.concatenate(returned_indices)

    assert np.allclose(expected_indices, np.sort(returned_indices))


@pytest.mark.parametrize("nr_samples, split_size, seed", [
    (100, (5, 0, 1), 1),
    (7885, (5, 0), 2),
    (354, (7, 3, 4), 3),
])
def test_split_kfold_seed(nr_samples, split_size, seed):
    """tests seed behavior of split_kfold"""
    from datasets.utils import split_kfold
    import numpy as np

    train_indices, val_indices, test_indices = split_kfold(nr_samples, split_size, seed=seed)

    returned_indices = [train_indices, val_indices]
    returned_indices += [test_indices] if test_indices is not None else []
    returned_indices = np.concatenate(returned_indices)

    # set seed globally
    np.random.seed(seed * 2)

    train_indices_again, val_indices_again, test_indices_again = split_kfold(nr_samples, split_size, seed=seed)

    returned_indices_again = [train_indices_again, val_indices_again]
    returned_indices_again += [test_indices_again] if test_indices_again is not None else []
    returned_indices_again = np.concatenate(returned_indices_again)

    assert np.allclose(returned_indices, returned_indices_again)

    #

    train_indices_diffseed, val_indices_diffseed, test_indices_diffseed = split_kfold(nr_samples, split_size, seed=seed*3)

    returned_indices_diffseed = [train_indices_diffseed, val_indices_diffseed]
    returned_indices_diffseed += [test_indices_diffseed] if test_indices_diffseed is not None else []
    returned_indices_diffseed = np.concatenate(returned_indices_diffseed)

    assert not np.allclose(returned_indices, returned_indices_diffseed)