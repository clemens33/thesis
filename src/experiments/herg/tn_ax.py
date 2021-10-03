import multiprocessing
import os
import random
import sys
from argparse import Namespace, ArgumentParser

from datasets import Hergophores
from experiments import TuneAx
from experiments.kfold import Kfold
from tn import train_tn, train_tn_kfold


def train_evaluate(args: Namespace):
    args.feature_size = args.decision_size * 2

    if hasattr(args, "weight_decay"):
        args.optimizer_params["weight_decay"] = args.weight_decay

    if hasattr(args, "decay_step"):
        args.scheduler_params["decay_step"] = args.decay_step

    if hasattr(args, "decay_rate"):
        args.scheduler_params["decay_rate"] = args.decay_rate

    if hasattr(args, "warmup_steps"):
        args.scheduler_params["warmup_steps"] = args.warmup_steps

    if "kfold" in args.split_type:
        kfold = Kfold(
            function=train_tn_kfold,
            args=args,
            **vars(args)
        )
        results = kfold.train()

        metric = results[args.objective_name]
        # additional_metric = results.get("test/rs-mean_aurocs", .0)
        # additional_metric = results.get("val/Accuracy", .0)
        additional_metric = .0

        metric += additional_metric
    else:
        results_test, results_val_best, results_val_last, *_ = train_tn(args)

        metric = results_val_best[args.objective_name]

    return metric


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""
    # kfold options
    args.track_metrics = [
        "val/loss",
        "val/AUROC",
        "val/Accuracy",
        "val/sparsity_mask",
        "test/loss",
        "test/AUROC",
        "test/Accuracy",
        "test/sparsity_mask",
    ]
    args.track_metrics += [
        # "test/mean/avg_score_pred_active",
        "test/mean/avg_score_pred_inactive",
        "test/mean/avg_score_pred_inactive/tabnet",
        "test/mean/avg_score_pred_inactive/integrated_gradients",
        "test/mean/avg_score_pred_inactive/saliency",
        "test/mean/avg_score_pred_inactive/saliency-absolute",
        "test/mean/avg_score_pred_inactive/input_x_gradient",

        "test/mean/avg_score_pred_active",
        "test/mean/avg_score_pred_active/tabnet",
        "test/mean/avg_score_pred_active/integrated_gradients",
        "test/mean/avg_score_pred_active/saliency",
        "test/mean/avg_score_pred_active/saliency-absolute",
        "test/mean/avg_score_pred_active/input_x_gradient",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "data_types": ["test"],
        "methods": [
            {"tabnet": {
                "postprocess": None
            }},
            {"integrated_gradients": {
                "n_steps": 50,
                "postprocess": None
            }},
            {"saliency": {
                "postprocess": None,
                "abs": False,
            }},
            # {"saliency-absolute": {
            #     "postprocess": None,
            #     "abs": True,
            # }},
            # {"input_x_gradient": {
            #     "postprocess": None
            # }},
        ],
        "track_metrics": args.track_metrics,
        "label": "active_g10",
        "label_idx": 0,
        "references": Hergophores.ACTIVES_UNIQUE_,
        # "nr_samples": 100,
    }

    # ax args
    args.trials = 25
    args.trials_sobol = 8
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.search_space = [
        #{"name": "batch_size", "type": "choice", "values": [64, 256, 512]},

        {"name": "decision_size", "type": "choice", "values": [8, 16, 32, 64]},
        {"name": "nr_steps", "type": "choice", "values": [3, 4, 5, 7]},
        {"name": "gamma", "type": "choice", "values": [1.0, 1.2, 1.5, 2.0]},
        {"name": "lambda_sparse", "type": "choice", "values": [0.0, 1e-6, 1e-4]},

        #{"name": "virtual_batch_size", "type": "choice", "values": [8, 16, 32]},
        #{"name": "momentum", "type": "choice", "values": [0.1, 0.03, 0.01]},

        # {"name": "lr", "type": "choice", "values": [0.005, 0.01, 0.02, 0.025]},
        {"name": "lr", "type": "range", "bounds": [0.0005, 0.03], "log_scale": False},
        {"name": "decay_step", "type": "choice", "values": [50, 200]},
        {"name": "decay_rate", "type": "choice", "values": [0.8, 0.9, 0.95]},

        #{"name": "lr", "type": "range", "bounds": [0.005, 0.05], "log_scale": True},
        #{"name": "lr", "type": "choice", "values": [0.005, 0.01, 0.02, 0.03, 0.05]},
        #{"name": "weight_decay", "type": "choice", "values": [0.0, 1e-4, 1e-3]},
        #{"name": "warmup_steps", "type": "choice", "values": [0.01, 0.1, 0.3]},

        # {"name": "decay_rate", "type": "range", "bounds": [0.0, 1.0]},
    ]

    # trainer/logging args
    args.experiment_name = "herg_tn_ax3009_01"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 1000
    args.seed = random.randint(0, 2 ** 32 - 1)
    args.checkpoint_objective = "val/loss"
    args.checkpoint_minimize = True
    args.patience_objective = "val/loss"
    args.patience_minimize = True
    args.patience = 10
    args.stochastic_weight_avg = False
    args.gradient_clip_val = 1.0

    # data module args
    args.batch_size = 256
    # args.split_type = "random_kfold"
    # args.split_size = (5, 0, 1)
    args.split_type = "random"
    args.split_size = (0.6, 0.2, 0.2)
    args.split_seed = random.randint(0, 2 ** 32 - 1)
    # args.use_labels = ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"]
    args.use_labels = ["active_g10"]
    args.standardize = True

    args.featurizer_name = "combined"  # ecfp + macc + tox
    args.featurizer_kwargs = {
        "fold": 1024,
        "radius": 3,
        "return_count": True,
        "use_chirality": True,
        "use_features": True,
    }

    # args.num_workers = multiprocessing.cpu_count()
    args.num_workers = 8

    args.cache_dir = "../../../" + "data/herg/"

    args.decision_size = 16
    args.feature_size = args.decision_size * 2
    args.nr_layers = 4
    args.nr_shared_layers = 2
    args.nr_steps = 3
    args.relaxation_type = "gamma_fixed"
    args.gamma = 1.5
    args.attentive_type = "sparsemax"
    #args.alpha = 2.0

    # args.lambda_sparse = 1e-6
    # args.lambda_sparse = 0.1
    args.lambda_sparse = 1e-6

    args.virtual_batch_size = -1  # -1 do not use any batch normalization
    args.normalize_input = False
    #args.virtual_batch_size = 16  # -1 do not use any batch normalization
    #args.momentum = 0.02

    args.optimizer = "adamw"
    args.optimizer_params = {"weight_decay": 0.0001}
    # args.lr = 0.01
    # args.optimizer = "adam"

    args.scheduler = "exponential_decay"
    args.scheduler_params = {"decay_step": 200, "decay_rate": 0.95}
    # args.scheduler = "linear_with_warmup"
    # args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params = {"warmup_steps": 0.1}

    args.log_sparsity = True
    # args.log_sparsity = "verbose"
    # args.log_parameters = False

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
