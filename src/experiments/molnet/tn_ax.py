import sys
import os
import random
from argparse import Namespace, ArgumentParser

from tn import train_tn
from experiments import TuneAx


def train_evaluate(args: Namespace):
    args.feature_size = args.decision_size * 2

    #args.scheduler_params["decay_step"] = args.decay_step
    #args.scheduler_params["decay_rate"] = args.decay_rate
    #args.scheduler_params["virtual_batch_size"] = args.batch_size

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
        #{"name": "batch_size", "type": "choice", "values": [32, 64, 128]},

        {"name": "decision_size", "type": "choice", "values": [24, 32, 64, 128]},
        {"name": "nr_steps", "type": "range", "bounds": [3, 8]},
        {"name": "gamma", "type": "choice", "values": [1.0, 1.2, 1.5, 2.0]},

        #{"name": "virtual_batch_size", "type": "choice", "values": [-1, 8, 32, 64, 512]},
        #{"name": "momentum", "type": "choice", "values": [0.1, 0.05, 0.02]},

        {"name": "lambda_sparse", "type": "choice", "values": [0.0, 1e-06, 0.0001, 0.001]},
        # {"name": "lr", "type": "choice", "values": [0.005, 0.01, 0.02, 0.025]},
        {"name": "lr", "type": "range", "bounds": [5e-05, 0.005], "log_scale": True},

        #{"name": "decay_step", "type": "choice", "values": [50, 200]},
        #{"name": "decay_rate", "type": "choice", "values": [0.8, 0.9, 0.95]},
        # {"name": "decay_rate", "type": "range", "bounds": [0.0, 1.0]},
    ]

    # trainer/logging args
    args.experiment_name = "hiv_tn_4096_ax4"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.gradient_clip_val = 1.0
    args.max_steps = 1000
    args.seed = 0
    args.patience = 50

    # data module args
    args.data_name = "hiv"
    args.batch_size = 2048
    args.split_seed = 0
    args.split_type = "random"
    args.n_bits = 4096
    args.radius = 4
    args.chirality = True
    args.features = True
    args.featurizer_name = "ecfp"

    args.num_workers = 4
    args.cache_dir = "../../../" + "data/molnet/hiv/"

    # model args
    args.decision_size = 64
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 6
    # args.alpha = 2.0
    args.relaxation_type = "gamma_fixed"
    #args.attentive_type = "softmax"
    #args.slope = 3.0
    #args.slope = 3.0
    # args.alpha_trainable = True
    args.gamma = 1.0
    # args.relaxation_type = "gamma_shared_trainable"
    # args.gamma_shared_trainable = True
    # args.gamma_trainable = True
    args.lambda_sparse = 0.0

    #args.virtual_batch_size = 256  # -1 do not use any batch normalization
    args.virtual_batch_size = -1
    #args.momentum = 0.1
    args.normalize_input = False
    #args.normalize_input = True

    args.lr = 0.001
    args.optimizer = "adam"
    #args.scheduler = "exponential_decay"
    #args.scheduler_params = {"decay_step": 100, "decay_rate": 0.95}

    #args.optimizer = "adamw"
    #args.optimizer_params = {"weight_decay": 0.0001}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    #args.scheduler_params = {"warmup_steps": 0.05}

    # args.index_embeddings = True
    # args.categorical_embeddings = True
    # args.categorical_indices = list(range(args.n_bits))
    # args.categorical_size = [2] * args.n_bits
    # args.embedding_dims = 1
    # args.embedding_dims = [1] * len(CovTypeDataModule.BINARY_COLUMNS)

    args.log_sparsity = True
    #args.log_parameters = True

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
