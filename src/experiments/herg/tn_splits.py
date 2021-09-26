import os
import sys
from argparse import Namespace, ArgumentParser

import numpy as np

from datasets import Hergophores
from experiments.grid_runner import GridRunner
from experiments.kfold import Kfold
from tn import train_tn, train_tn_kfold


def train(args: Namespace, **kwargs):
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
        metrics = kfold.train()

    else:
        results_test, results_val_best, results_val_last, results_attribution, *_ = train_tn(args)

        metrics = {}
        for metric_name in args.track_metrics:
            if metric_name in results_test:
                metrics[metric_name] = results_test[metric_name]
            if metric_name in results_val_best:
                metrics[metric_name] = results_val_best[metric_name]
            elif metric_name in results_attribution:
                metrics[metric_name] = results_attribution[metric_name]

    return metrics


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""
    # grid runner args
    args.max_trials = 9999999

    args.track_metrics = [
        "val/AUROC",
        "val/Accuracy",
        "test/AUROC",
        "test/Accuracy",
    ]
    args.track_metrics += [
        # "test/mean/avg_score_pred_active",
        "test/mean/avg_score_pred_inactive",
        "test/mean/avg_score_pred_inactive/integrated_gradients",
        "test/mean/avg_score_pred_inactive/saliency",
        "test/mean/avg_score_pred_inactive/input_x_gradient",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "data_types": ["test"],
        "methods": [
            {"default": {
                "postprocess": None
            }},
            {"integrated_gradients": {
                "postprocess": None
            }},
            {"saliency": {
                "postprocess": None,
                "abs": False,  # Returns absolute value of gradients if set to True
            }},
            {"input_x_gradient": {
                "postprocess": None
            }},
        ],
        "track_metrics": args.track_metrics,
        "label": "active_g10",
        "label_idx": 0,
        "references": Hergophores.ACTIVES_UNIQUE_,
        # "nr_samples": 100,
    }

    # seeds
    args.seed = 0
    args.split_seed = 1

    args.grid_space = {
        "seed": np.random.default_rng(args.seed).integers(2 ** 32 - 1, size=(20,)).tolist(),
        # "split_seed": np.random.default_rng(args.seed + 1).integers(2 ** 32 - 1, size=(10,)).tolist(),
        #"seed": [159, 753, 951, 456, 852, 789, 123, 147, 258, 369]
    }

    # trainer/logging args
    args.experiment_name = "herg_tn_diff_seed3"
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.run_name = "diff_seeds2"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 1000
    args.patience = 10
    args.stochastic_weight_avg = True

    # data module args
    args.batch_size = 64
    # args.split_type = "random_kfold"
    # args.split_size = (5, 0, 1)
    args.split_type = "random"
    args.split_size = (0.6, 0.2, 0.2)
    # args.use_labels = ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"]
    args.use_labels = ["active_g10"]

    args.featurizer_name = "combined"  # ecfp + macc + tox
    args.featurizer_kwargs = {
        "fold": 1024,
        "radius": 3,
        "return_count": True,
        "use_chirality": True,
        "use_features": True,
    }
    args.featurizer_mp_context = "fork"
    args.featurizer_chunksize = 100

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/herg/"

    # model args
    args.decision_size = 32
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 7
    args.gamma = 1.2

    args.relaxation_type = "gamma_fixed"
    # args.alpha = 2.0
    # args.attentive_type = "sparsemax"

    # args.slope = 3.0
    # args.slope_type = "slope_fixed"

    # args.lambda_sparse = 1e-6
    # args.lambda_sparse = 0.1
    args.lambda_sparse = 1e-06

    # args.virtual_batch_size = 32  # -1 do not use any batch normalization
    args.virtual_batch_size = -1  # -1 do not use any batch normalization

    # args.normalize_input = True
    # args.virtual_batch_size = 64
    # args.momentum = 0.1

    args.normalize_input = False
    # args.virtual_batch_size = 256  # -1 do not use any batch normalization

    args.lr = 0.001
    # args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    args.optimizer = "adamw"
    args.optimizer_params = {"weight_decay": 0.001}
    args.scheduler = "linear_with_warmup"
    # args.scheduler_params = {"warmup_steps": 10}
    args.scheduler_params = {"warmup_steps": 0.05}

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

    runner = GridRunner(
        function=train,
        args=args,

        **vars(args)
    )
    metrics = runner.run()

    print(metrics)
