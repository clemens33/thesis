import sys
import os
import random
from argparse import Namespace, ArgumentParser

from tn import train_tn
from experiments import TuneAx


def train_evaluate(args: Namespace):
    args.feature_size = args.decision_size * 2

    # args.scheduler_params["decay_step"] = args.decay_step
    # args.scheduler_params["decay_rate"] = args.decay_rate

    results_test, results_val_best, results_val_last, *_ = train_tn(args)

    metric = results_val_best[args.objective_name]
    # metric = results_val_last[args.objective_name]
    # metric= random.random()

    return metric


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # ax args
    args.trials = 25
    args.trials_sobol = 10
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.search_space = [
        # {"name": "batch_size", "type": "choice", "values": [128, 256, 512, 1024]},

        {"name": "decision_size", "type": "choice", "values": [8, 16, 24, 32, 64, 128]},
        {"name": "nr_steps", "type": "range", "bounds": [3, 10]},
        {"name": "gamma", "type": "choice", "values": [1.0, 1.2, 1.5, 2.0]},

        # {"name": "virtual_batch_size", "type": "choice", "values": [-1, 8, 32, 64, 512]},
        # {"name": "momentum", "type": "choice", "values": [0.4, 0.3, 0.2, 0.1, 0.05, 0.02]},

        {"name": "lambda_sparse", "type": "choice", "values": [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1]},
        # {"name": "lr", "type": "choice", "values": [0.005, 0.01, 0.02, 0.025]},
        {"name": "lr", "type": "range", "bounds": [1e-5, 0.01], "log_scale": True},

        # {"name": "decay_step", "type": "choice", "values": [50, 200, 800]},
        # {"name": "decay_rate", "type": "choice", "values": [0.4, 0.8, 0.9, 0.95]},
        # {"name": "decay_rate", "type": "range", "bounds": [0.0, 1.0]},
    ]

    # trainer/logging args
    args.experiment_name = "bbbp_tn_1024_4_ax2"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 2000
    args.seed = 0
    args.patience = 100

    # data module args
    args.data_name = "bbbp"
    args.batch_size = 256
    args.split_seed = 0
    args.n_bits = 1024
    args.radius = 4
    args.chirality = True
    args.features = True
    args.featurizer_name = "ecfp"

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.decision_size = 24
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 6
    args.alpha = -2.0
    # args.alpha_trainable = True
    args.gamma = 1.5
    # args.gamma_shared_trainable = True
    # args.gamma_trainable = True
    args.lambda_sparse = 1e-6

    args.virtual_batch_size = 256  # -1 do not use any batch normalization
    # args.virtual_batch_size = -1
    args.momentum = 0.1
    # args.normalize_input = False
    args.normalize_input = True

    args.lr = 0.001
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 100, "decay_rate": 0.95}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.0001}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params = {"warmup_steps": 0.1}

    # args.index_embeddings = True
    # args.categorical_embeddings = True
    # args.categorical_indices = list(range(args.n_bits))
    # args.categorical_size = [2] * args.n_bits
    # args.embedding_dims = 1
    # args.embedding_dims = [1] * len(CovTypeDataModule.BINARY_COLUMNS)

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

    ax = TuneAx(
        function=train_evaluate,
        args=args,

        **vars(args)
    )
    best_parameters, values = ax.optimize()

    print(best_parameters)
    print(values)
