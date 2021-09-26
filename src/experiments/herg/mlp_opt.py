import sys
import os
import random
from argparse import Namespace, ArgumentParser

from datasets import Hergophores
from experiments.kfold import Kfold
from mlp import train_mlp, train_mlp_kfold
from experiments.tune_optuna import TuneOptuna


def train_evaluate(args: Namespace, **kwargs):
    if hasattr(args, "nr_layers") and isinstance(args.hidden_size, int):
        args.hidden_size = [args.hidden_size] * args.nr_layers

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
            function=train_mlp_kfold,
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
        results_test, results_val_best, results_val_last, *_ = train_mlp(args)

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
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "data_types": ["test"],
        "methods": [
            {"integrated_gradients": {
                "n_steps": 50,
                "postprocess": None
            }},
            {"saliency": {
                "postprocess": None,
                "abs": False,
            }},
            {"saliency-absolute": {
                "postprocess": None,
                "abs": True,
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

    # optuna args
    args.trials = 30
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.sampler_name = "tpe"
    args.pruner_name = None
    args.search_space = [
        {"name": "batch_size", "type": "choice", "values": [32, 64, 128, 256, 512]},

        {"name": "hidden_size", "type": "choice", "values": [16, 32, 64, 128]},
        {"name": "nr_layer", "type": "choice", "values": [1, 2, 3, 4, 5]},

        {"name": "dropout", "type": "choice", "values": [0.0, 0.1, 0.01, 0.3]},
        {"name": "momentum", "type": "choice", "values": [0.2, 0.1, 0.05, 0.01]},

        {"name": "lr", "type": "choice", "values": [0.02, 0.01, 0.05, 0.001]},
        {"name": "warmup_steps", "type": "choice", "values": [0.01, 0.05, 0.1, 0.3]},
        {"name": "weight_decay", "type": "choice", "values": [0.0, 0.001, 0.0001]},
        # {"name": "lr", "type": "range", "bounds": [1e-5, 0.01], "log_scale": True},

        # {"name": "decay_step", "type": "choice", "values": [50, 200, 800]},
        # {"name": "decay_rate", "type": "choice", "values": [0.4, 0.8, 0.9, 0.95]},
        # {"name": "decay_rate", "type": "range", "bounds": [0.0, 1.0]},
    ]

    # trainer/logging args
    args.experiment_name = "herg_mlp_opttpe1"
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
    args.standardize = False

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

    args.run_name = "mlp random"

    # model args
    args.hidden_size = [32] * 3
    args.dropout = 0.1
    args.normalize_input = True
    args.batch_norm = True
    args.momentum = 0.01

    # scheduler
    args.lr = 0.01
    args.optimizer = "adam"
    args.scheduler = "exponential_decay"
    args.scheduler_params = {"decay_step": 10, "decay_rate": 0.95}

    args.optimizer = "adamw"
    args.optimizer_params = {"weight_decay": 0.001}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 0.1}

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

    opt = TuneOptuna(
        function=train_evaluate,
        args=args,

        **vars(args)
    )
    best_parameters, value = opt.optimize()

    print(best_parameters)
    print(value)
