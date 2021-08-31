import math
from typing import Tuple, Union

import numpy as np
from numpy.random import default_rng


def add_noise_features(input: np.ndarray, factor: float, type="standard_normal", position: str = "random", seed: int = 0) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    adds random/noise features along last dimension/axis

    Args:
        input (): random features will be added to input along last dimension
        factor (): determines how many random features will be added - if input feature size is 6 and factor is 1.0 we add 6 random features.
        If factor is 0.5 we add 3, if 2.0 we add 12 random features
        type (): type of random features added - supports zeros, ones, binary or standard normal
        position (): determines where in the output the random features will be located - supports left, right or random
        seed ():

    Returns:
        - input (np.ndarray): input added with random features
        - feature_indices (np.ndarray): real feature indices locations

    """
    assert factor > 0.0, f"factor {factor} must be greater than 0"

    nr_features = input.shape[-1]
    nr_features_noise = int(nr_features * factor)
    nr_features_new = nr_features + nr_features_noise
    new_shape = input.shape[:-1] + (nr_features_new,)

    rng = default_rng(seed)

    # initialize noise/random features
    if type == "zeros":
        input_new = np.zeros(shape=new_shape, dtype=input.dtype)
    elif type == "ones":
        input_new = np.ones(shape=new_shape, dtype=input.dtype)
    elif type == "binary":
        np.random.seed(seed)
        input_new = rng.integers(2, size=new_shape).astype(dtype=input.dtype)
    elif type == "standard_normal":
        input_new = rng.standard_normal(size=new_shape).astype(dtype=input.dtype)
    elif type == "replicate":
        f = math.ceil(factor) + 1
        input_new = np.tile(input, f)[..., :nr_features_new]
    else:
        raise ValueError(f"noise feature type {type} is not implemented.")

    # add noise/random features at defined position
    if position == "right":
        feature_indices = np.arange(nr_features)
    elif position == "left":
        feature_indices = np.arange(nr_features_new - nr_features, nr_features_new)
    elif position == "random":
        feature_indices = rng.choice(np.arange(nr_features_new), size=nr_features, replace=False)
    else:
        raise ValueError(f"noise feature position {position} is not implemented.")

    input_new[..., feature_indices] = input
    return input_new, feature_indices


def add_noise(input: np.ndarray, type: str = "standard_normal", seed: int = 0) -> np.ndarray:
    """adds noise to input"""

    rng = default_rng(seed)

    if type == "standard_normal":
        noise = rng.standard_normal(size=input.shape).astype(dtype=input.dtype)
    elif type == "zeros_standard_normal":
        input = input * 10000

        noise = rng.standard_normal(size=input.shape).astype(dtype=input.dtype)
        input = input + noise

        input[input > 900] = 1

        return input
    elif type == "zeros_standard_normal2":
        input = input * 10000

        noise = rng.standard_normal(size=input.shape).astype(dtype=input.dtype)
        input = input + noise

        input[input > 900] = 100

        return input
    elif type == "replace_zeros":
        input[input == 0] = -1

        return input
    elif type == "add_ones":
        noise = np.ones_like(input)
    else:
        noise = np.zeros_like(input)

    return input + noise


def split_kfold(nr_samples: int, split_size: Tuple[int, ...] = (5, 0, 1), seed: int = 0) -> Tuple[
    np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    rng = np.random.default_rng(seed)

    if len(split_size) < 2:
        raise ValueError(f"split size {split_size} must contain at least nr of folds and the validation fold to use")
    if not all([f < split_size[0] for f in split_size[1:]]):
        raise ValueError(f"defined validation or test folds to use must be zero indexed and not exceed the total nr of folds")

    indices = np.arange(nr_samples)
    rng.shuffle(indices)

    nr_folds = split_size[0]
    folds = np.array_split(indices, nr_folds)

    val_fold_nr = split_size[1]
    test_fold_nr = split_size[2] if len(split_size) > 2 else -1
    train_fold_nrs = [f for f in range(nr_folds) if f not in [val_fold_nr, test_fold_nr]]

    train_indices = np.concatenate([folds[f] for f in train_fold_nrs])
    val_indices = folds[val_fold_nr]
    test_indices = folds[test_fold_nr] if test_fold_nr != -1 else None

    return train_indices, val_indices, test_indices
