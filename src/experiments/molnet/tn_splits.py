import random
import sys
from argparse import Namespace, ArgumentParser
from typing import Dict

import mlflow
import numpy as np
from pytorch_lightning import seed_everything

from tn import train_tn
from experiments.splits import splits


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # test args
    args.trials = 20
    args.split_seed_init = 0

    # trainer/logging args
    args.experiment_name = "bbbp_tn_ecfc_test1"
    args.tracking_uri = "https://mlflow.kriechbaumer.at"
    args.max_steps = 1000
    args.seed = 0  # model seed
    args.patience = 50

    # data module args
    args.data_name = "bbbp"
    args.batch_size = 256
    args.split_seed = 0
    #args.n_bits = 4096
    args.radius = 6
    args.chirality = True
    args.features = True
    args.featurizer_name = "ecfc"

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.decision_size = 16
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 3
    args.gamma = 1.0
    args.lambda_sparse = 0.0

    args.virtual_batch_size = 32  # -1 do not use any batch normalization
    args.normalize_input = True
    args.momentum = 0.3

    args.lr = 0.005
    args.optimizer = "adam"
    args.scheduler = "exponential_decay"
    args.scheduler_params = {"decay_step": 800, "decay_rate": 0.8}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.00005}
    #args.scheduler = "linear_with_warmup"
    #args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params={"warmup_steps": 0.01}

    args.categorical_embeddings = True

    #args.log_sparsity = True

    return args


def run_cli() -> Namespace:
    parser = ArgumentParser()

    # parser.add_argument("--test", type=bool, default=True)
    args = parser.parse_args()

    # if no arguments have been provided we use manually set arguments - for debugging/dev
    args = manual_args(args) if len(sys.argv) <= 1 else args

    return args


if __name__ == "__main__":
    args = run_cli()

    results = splits(train_tn, args)

    print(results)
