import sys
import os
import random
from argparse import Namespace, ArgumentParser

from datasets import Hergophores
from experiments.kfold import Kfold
from gbdt import train_gbdt, train_gbdt_kfold
from experiments.tune_optuna import TuneOptuna


def train_evaluate(args: Namespace, **kwargs):
    if hasattr(args, "n_estimators"):
        args.gbdt_kwargs["n_estimators"] = args.n_estimators

    if hasattr(args, "learning_rate"):
        args.gbdt_kwargs["learning_rate"] = args.learning_rate

    if hasattr(args, "min_child_weight"):
        args.gbdt_kwargs["min_child_weight"] = args.min_child_weight

    if hasattr(args, "max_depth"):
        args.gbdt_kwargs["max_depth"] = args.max_depth

    if hasattr(args, "gamma"):
        args.gbdt_kwargs["gamma"] = args.gamma

    if hasattr(args, "colsample_bytree"):
        args.gbdt_kwargs["colsample_bytree"] = args.colsample_bytree

    if hasattr(args, "subsample"):
        args.gbdt_kwargs["subsample"] = args.subsample

    if hasattr(args, "max_delta_step"):
        args.gbdt_kwargs["max_delta_step"] = args.max_delta_step

    if "kfold" in args.split_type:
        kfold = Kfold(
            function=train_gbdt_kfold,
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
        results_test, results_val_best, results_val_last, *_ = train_gbdt(args)

        metric = results_val_best[args.objective_name]

    return metric


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""
    # kfold options
    args.track_metrics = [
        "val/AUROC",
        "val/Accuracy",

        "test/AUROC",
        "test/Accuracy",
    ]
    args.track_metrics += [
        "test/mean/avg_score_pred_inactive/feature_importances",
        "test/mean/avg_score_pred_inactive/shap",

        "test/mean/avg_score_pred_active/feature_importances",
        "test/mean/avg_score_pred_active/shap",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "data_types": ["test"],
        "methods": [
            {"feature_importances": {
                "postprocess": None
            }},
            {"shap": {
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
        {"name": "n_estimators", "type": "choice", "values": [10, 50, 100, 200, 500]},
        {"name": "learning_rate", "type": "choice", "values": [0.3, 0.1, 0.05, 0.01]},
        {"name": "min_child_weight", "type": "choice", "values": [1, 5, 10]},
        {"name": "max_depth", "type": "choice", "values": [None] + list(range(4, 40, 4))},
        {"name": "gamma", "type": "choice", "values": [0.5, 1, 1.5, 2, 5]},
        {"name": "colsample_bytree", "type": "choice", "values": [0.6, 0.8, 1.0]},
        {"name": "subsample", "type": "choice", "values": [0.6, 0.8, 1.0]},
        {"name": "max_delta_step", "type": "choice", "values": [0, 0.1, 1, 10]},
    ]

    # trainer/logging args
    args.experiment_name = "herg_gbdt_opttpe1"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.seed = random.randint(0, 2 ** 32 - 1)
    args.patience = 20

    # data module args
    args.batch_size = 9999
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

    args.num_workers = 0
    args.cache_dir = "../../../" + "data/herg/"

    args.run_name = "tpe"

    # model args
    args.gbdt_kwargs = {
        # "booster": "gbtree",
        "n_estimators": 100,
        "learning_rate": 0.3,
        "gamma": 0.0,  # [0.0, 0.1, 0.2, 0.3, 0.4]
        "max_depth": 6,
        "min_child_weight": 1,
        "max_delta_step": 0,
        "colsample_bytree": 0.3,  # [0.3, 0.4, 0.5, 0.7]
    }

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
