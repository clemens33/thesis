import os
import sys
from argparse import Namespace, ArgumentParser

from experiments.splits import splits
from tn import train_tn


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # test args
    args.trials = 20
    args.split_seed_init = 2

    # trainer/logging args
    args.experiment_name = "bbbp_tn_ecfc-6_GST_test3"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 3000
    args.seed = 2  # model seed
    args.patience = 9999999

    # data module args
    args.data_name = "bbbp"
    args.batch_size = 256
    args.split_seed = 2
    # args.n_bits = 4096
    args.radius = 6
    args.chirality = True
    args.features = True
    args.featurizer_name = "ecfc"

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.decision_size = 64
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 3
    args.alpha = -2.0
    args.gamma = 1.5
    args.gamma_shared_trainable = True
    args.lambda_sparse = 0.1

    args.virtual_batch_size = 256  # -1 do not use any batch normalization
    # args.virtual_batch_size = -1  # -1 do not use any batch normalization
    args.normalize_input = True
    args.momentum = 0.1

    args.lr = 0.009994961103457224
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 200, "decay_rate": 0.95}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.00005}
    args.scheduler = "linear_with_warmup"
    # args.scheduler_params = {"warmup_steps": 10}
    args.scheduler_params = {"warmup_steps": 0.1}

    args.categorical_embeddings = True
    args.embedding_dims = 1

    # args.log_sparsity = True
    args.log_sparsity = True
    args.log_parameters = True

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
