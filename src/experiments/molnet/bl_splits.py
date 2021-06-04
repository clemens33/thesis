import os
import sys
from argparse import Namespace, ArgumentParser

from bl import train_bl
from experiments.splits import splits


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # splits args
    args.trials = 20
    args.split_seed_init = 0

    # trainer/logging args
    args.experiment_name = "bbbp_bl_ecfc_nonorm_noemb_test1"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 1000
    args.seed = 0  # model seed
    args.patience = 50

    # data module args
    args.data_name = "bbbp"
    args.batch_size = 256
    args.split_seed = 0
    # args.n_bits = 4096
    args.radius = 6
    args.chirality = True
    args.features = True
    args.featurizer_name = "ecfc"

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.hidden_size = [188] * 3
    args.dropout = 0.3

    args.lr = 2.474e-4
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.00005}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params={"warmup_steps": 0.01}

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

    results = splits(train_bl, args)

    print(results)
