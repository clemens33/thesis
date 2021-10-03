import sys
import os
import random
from argparse import Namespace, ArgumentParser

from datasets import Hergophores
from experiments.kfold import Kfold
from rf import train_rf, train_rf_kfold
from experiments.tune_optuna import TuneOptuna


def train_evaluate(args: Namespace, **kwargs):
    if hasattr(args, "n_estimators"):
        args.rf_kwargs["n_estimators"] = args.n_estimators

    if hasattr(args, "max_features"):
        args.rf_kwargs["max_features"] = args.max_features

    if hasattr(args, "criterion"):
        args.rf_kwargs["criterion"] = args.criterion

    if hasattr(args, "max_depth"):
        args.rf_kwargs["max_depth"] = args.max_depth

    if hasattr(args, "min_samples_split"):
        args.rf_kwargs["min_samples_split"] = args.min_samples_split

    if hasattr(args, "min_samples_leaf"):
        args.rf_kwargs["min_samples_leaf"] = args.min_samples_leaf

    if hasattr(args, "bootstrap"):
        args.rf_kwargs["bootstrap"] = args.bootstrap

    if "kfold" in args.split_type:
        kfold = Kfold(
            function=train_rf_kfold,
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
        results_test, results_val_best, results_val_last, *_ = train_rf(args)

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

        "test/mean/avg_score_pred_inactive/impurity",
        "test/mean/avg_score_pred_inactive/treeinterpreter",
        "test/mean/avg_score_pred_inactive/permutation",
        "test/mean/avg_score_pred_inactive/input_x_impurity",
        "test/mean/avg_score_pred_inactive/occlusion",
        "test/mean/avg_score_pred_inactive/shapley_value_sampling",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "data_types": ["test"],
        "methods": [
            {"impurity": {
                "postprocess": None
            }},
            {"treeinterpreter": {
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
        {"name": "max_features", "type": "choice", "values": ["auto", "sqrt"]},
        {"name": "criterion", "type": "choice", "values": ["gini", "entropy"]},
        {"name": "max_depth", "type": "choice", "values": [None] + list(range(10, 80, 10))},
        {"name": "min_samples_split", "type": "choice", "values": [2, 5, 10]},
        {"name": "min_samples_leaf", "type": "choice", "values": [1, 2, 4]},
        {"name": "bootstrap", "type": "choice", "values": [True, False]},
    ]

    # trainer/logging args
    args.experiment_name = "herg_rf_opttpe1"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.seed = random.randint(0, 2 ** 32 - 1)

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
    args.rf_kwargs = {
        "n_estimators": 100,
        # "max_features": None
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
