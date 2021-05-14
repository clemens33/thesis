import sys
from argparse import Namespace, ArgumentParser
from typing import Dict

from ax import Models
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.service.managed_loop import optimize

from pytorch_lightning import seed_everything

from bbbp_tn import train_tn


def train_evaluate(parameterization: Dict):
    args = global_args

    # overwrite all arguments by the search space
    for key, value in parameterization.items():
        if key != "args":
            setattr(args, key, value)

    args.feature_size = args.decision_size * 2
    results_val, *_ = train_tn(args)

    # metric = trainer.callback_metrics["val/AUROC"].item()
    metric = results_val[0]["val/AUROC"]

    return metric


def ax_optimize(args: Namespace):
    global global_args  # TODO find better solution than a global variable
    global_args = args

    generation_strategy = GenerationStrategy(name="Sobol+GPEI",
                                             steps=[
                                                 GenerationStep(model=Models.SOBOL, num_trials=5),
                                                 GenerationStep(model=Models.GPEI, num_trials=-1)]
                                             )

    best_parameters, values, experiment, model = optimize(
        parameters=args.search_space,
        total_trials=args.trials,
        random_seed=args.seed_init,
        evaluation_function=train_evaluate,
        objective_name="val/AUROC",
        minimize=False,
        generation_strategy=generation_strategy
    )

    return best_parameters, values, experiment, model


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # ax args
    args.trials = 20
    args.seed_init = 0

    # trainer/logging args
    args.experiment_name = "bbbp_tn_ax_32768"
    args.tracking_uri = "https://mlflow.kriechbaumer.at"
    args.max_steps = 1000
    args.seed = 0
    args.patience = 20

    # data module args
    args.batch_size = 128
    args.split_seed = 0
    args.n_bits = 32768
    args.radius = 4
    args.chirality = True
    args.features = True

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.decision_size = 64
    args.feature_size = 256
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 10
    args.gamma = 1.8
    args.lambda_sparse = 1e-6

    args.virtual_batch_size = -1  # -1 do not use any batch normalization
    args.normalize_input = False

    args.lr = 0.001
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    #args.optimizer = "adamw"
    #args.optimizer_params = {"weight_decay": 0.0001}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params={"warmup_steps": 0.01}

    # ax parameters - will overwrite default args if available

    # args.search_space = [
    #     {"name": "batch_size", "type": "choice", "values": [128, 256, 512, 1024, 2048]},
    #     {"name": "decision_size", "type": "choice", "values": [8, 16, 24, 32, 64, 128]},
    #     {"name": "nr_steps", "type": "choice", "values": [3, 4, 5, 6, 7, 8, 9, 10]},
    #     {"name": "gamma", "type": "choice", "values": [1.0, 1.2, 1.5, 2.0]},
    #     {"name": "lambda_sparse", "type": "choice", "values": [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1]},
    #     {"name": "lr", "type": "choice", "values": [0.005, 0.01, 0.02, 0.025]},
    #     {"name": "decay_step", "type": "choice", "values": [5, 20, 80, 100]},
    #     {"name": "decay_rate", "type": "choice", "values": [0.4, 0.8, 0.9, 0.95]},
    # ]

    args.search_space = [
        {"name": "batch_size", "type": "choice", "values": [128, 256, 512, 1024]},

        {"name": "decision_size", "type": "range", "bounds": [8, 64]},
        {"name": "nr_steps", "type": "range", "bounds": [3, 10]},
        {"name": "gamma", "type": "choice", "values": [1.0, 1.2, 1.5, 2.0]},

        {"name": "lambda_sparse", "type": "choice", "values": [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1]},
        {"name": "lr", "type": "range", "bounds": [1e-4, 0.025], "log_scale": True},

        # {"name": "decay_step", "type": "choice", "values": [5, 20, 80, 100]},
        # {"name": "decay_rate", "type": "choice", "values": [0.4, 0.8, 0.9, 0.95]},
    ]
    # args.search_space = [
    #     {"name": "batch_size", "type": "choice", "values": [128, 256, 512, 1024, 2048]},
    #
    #     {"name": "decision_size", "type": "choice", "values": [8, 16, 24, 32, 64, 128]},
    #     {"name": "nr_steps", "type": "choice", "values": [3, 4, 5, 6, 7, 8, 9, 10]},
    #     {"name": "gamma", "type": "choice", "values": [1.0, 1.2, 1.5, 2.0]},
    #
    #     {"name": "lambda_sparse", "type": "choice", "values": [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1]},
    #     {"name": "lr", "type": "choice", "values": [0.005, 0.01, 0.02, 0.025]},
    #     #{"name": "decay_step", "type": "choice", "values": [5, 20, 80, 100]},
    #     #{"name": "decay_rate", "type": "choice", "values": [0.4, 0.8, 0.9, 0.95]},
    # ]

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

    best_parameters, values, experiment, model = ax_optimize(args)

    print(best_parameters)
    print(values)
