import os
import sys
from argparse import Namespace, ArgumentParser

from experiments.splits import splits
from tn import train_tn


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # test args
    args.trials = 20
    args.split_seed_init = 0

    # trainer/logging args
    args.experiment_name = "hiv_tn_4096_test4"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.gradient_clip_val = 1.0
    args.max_steps = 1000
    args.seed = 0  # model seed
    args.patience = 50

    # data module args
    args.data_name = "hiv"
    args.batch_size = 2048
    args.split_type = "random"
    args.split_seed = 0
    args.n_bits = 4096
    args.radius = 4
    args.chirality = True
    args.features = True
    args.featurizer_name = "ecfp"

    args.num_workers = 4
    args.cache_dir = "../../../" + "data/molnet/hiv/"

    # model args
    args.decision_size = 128
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 3
    # args.alpha = 2.0
    # args.attentive_type = "sparsemax"
    # args.slope = 3.0
    args.gamma = 1.2
    # args.relaxation_type = "gamma_shared_trainable"
    args.lambda_sparse = 0.0
    # args.lambda_sparse = 0.0

    # args.virtual_batch_size = 256  # -1 do not use any batch normalization
    args.virtual_batch_size = -1  # -1 do not use any batch normalization
    args.normalize_input = False
    # args.momentum = 0.1

    args.lr = 0.00044816616909224065
    args.optimizer = "adam"
    #args.scheduler = "exponential_decay"
    #args.scheduler_params = {"decay_step": 200, "decay_rate": 0.8}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.00005}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params = {"warmup_steps": 10}

    # args.categorical_embeddings = True
    # args.embedding_dims = 1

    # args.log_sparsity = True
    args.log_sparsity = True
    args.log_parameters = False

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
