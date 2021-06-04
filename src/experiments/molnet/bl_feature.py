import sys
import os
from argparse import Namespace, ArgumentParser
from experiments.grid_runner import GridRunner
from bl import train_bl


def run(args):
    gs = GridRunner(function=train_bl, objective_name="test/AUROC", args=args, **vars(args))
    gs.run()


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # grid runner args
    # args.max_trials = 20
    args.grid_space = {
        "noise_features": [
            None,
            {
                "type": "zeros",
                "factor": 10.0,
                "position": "random",
            },
            {
                "type": "ones",
                "factor": 10.0,
                "position": "random",
            },
            {
                "type": "standard_normal",
                "factor": 10.0,
                "position": "random",
            },
            {
                "type": "replicate",
                "factor": 10.0,
                "position": "random",
            },
            {
                "type": "zeros",
                "factor": 10.0,
                "position": "right",
            },
            {
                "type": "zeros",
                "factor": 10.0,
                "position": "left",
            },
            {
                "type": "ones",
                "factor": 10.0,
                "position": "right",
            },
            {
                "type": "ones",
                "factor": 10.0,
                "position": "left",
            },
            {
                "type": "replicate",
                "factor": 10.0,
                "position": "right",
            },
            {
                "type": "replicate",
                "factor": 10.0,
                "position": "left",
            },
        ]
    }
    # args.grid_space = {
    #     "noise": [
    #         "standard_normal",
    #         "zeros_standard_normal",
    #         "zeros_standard_normal2",
    #         "replace_zeros",
    #         "add_ones"
    #     ]
    # }

    # trainer/logging args
    args.experiment_name = "bbbp_bl_random_features_512"
    args.tracking_uri=os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 1000
    args.seed = 0  # model seed
    args.patience = 50

    # data module args
    args.data_name = "bbbp"
    args.batch_size = 256
    args.split_seed = 0
    args.n_bits = 512
    args.radius = 4
    args.chirality = True
    args.features = True

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.hidden_size = [167] * 4
    args.dropout = 0.1

    args.lr = 1.818e-05
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.00005}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params={"warmup_steps": 0.01}

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

    run(args)
