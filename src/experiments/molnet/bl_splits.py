import os
import sys
from argparse import Namespace, ArgumentParser

from bl import train_bl
from experiments.splits import splits


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # test args
    args.trials = 20
    args.split_seed_init = 0

    # trainer/logging args
    args.experiment_name = "sider_bl_4096_test1"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 1000
    args.seed = 0  # model seed
    args.patience = 100

    # data module args
    args.data_name = "sider"
    args.batch_size = 256
    args.split_seed = 0
    args.n_bits = 4096
    args.radius = 4
    args.chirality = True
    args.features = True
    args.featurizer_name = "ecfp"

    args.num_workers = 4
    args.cache_dir = "../../../" + "data/molnet/sider/"

    # model args
    args.hidden_size = [189] * 4
    args.dropout = 0.3

    args.lr = 0.0005169184713279465
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.00005}
    args.scheduler = "linear_with_warmup"
    # args.scheduler_params = {"warmup_steps": 10}
    args.scheduler_params = {"warmup_steps": 10}

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
