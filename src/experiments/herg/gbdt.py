import os
import random
import sys
from argparse import Namespace, ArgumentParser
from pprint import pprint
from typing import Tuple, Dict

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import MLFlowLogger

from baseline.gbdt import GBDT
from datasets import Hergophores, HERGClassifierDataModule
from experiments.herg.attribution import Attribution
from experiments.kfold import Kfold


def train_gbdt_kfold(args: Namespace, split_size: Tuple[int, ...], split_seed: int) -> Dict[str, float]:
    """helper function to be called from Kfold implementation"""

    args.split_size = split_size
    args.split_seed = split_seed

    results_train_best, results_val_best, results_test_best, results_attribution, *_ = train_gbdt(args)

    metrics = {}
    for metric_name in args.track_metrics:
        if metric_name in results_test_best:
            metrics[metric_name] = results_test_best[metric_name]
        if metric_name in results_val_best:
            metrics[metric_name] = results_val_best[metric_name]
        elif metric_name in results_attribution:
            metrics[metric_name] = results_attribution[metric_name]

    return metrics


def train_gbdt(args: Namespace):
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=args.run_name if hasattr(args, "run_name") else None,
        tracking_uri=args.tracking_uri,
        tags=args.tags if hasattr(args, "tags") else None
    )

    dm = HERGClassifierDataModule(
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        split_type=args.split_type,
        split_size=args.split_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        use_labels=args.use_labels,
        use_cache=True,
        featurizer_name=args.featurizer_name,
        featurizer_kwargs=args.featurizer_kwargs,
        featurizer_mp_context="fork",
        featurizer_chunksize=100,
        standardize=args.standardize if hasattr(args, "standardize") else True,
    )
    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    seed_everything(args.seed)

    # workaround to ignore not labeled data
    if len(args.use_labels) > 1:
        class_weight = [{0: 1, 1: 1}]
        class_weight += [{-100: 0, 0: 1, 1: 1} for _ in range(1, len(args.use_labels))]
    else:
        class_weight = None

    gbdt = GBDT(
        input_size=dm.input_size,
        num_classes=dm.num_classes,
        seed=args.seed,
        class_weight=class_weight,
        gbdt_kwargs=args.gbdt_kwargs,
        logger=mlf_logger,
        patience=args.patience,
    )
    results_train_best = gbdt.fit(dm.train_dataloader(), dm.val_dataloader())
    results_val_best = gbdt.test(dm.val_dataloader(), stage="val")
    results_test_best = gbdt.test(dm.test_dataloader(), stage="test")

    results_attribution = {}
    if args.attribution_kwargs is not None:
        if "threshold" not in args.attribution_kwargs:
            key = "train/threshold-t" + str(args.attribution_kwargs["label_idx"])
            args.attribution_kwargs["threshold"] = results_train_best[key] if key in results_train_best else .5

        attributor = Attribution(
            model=gbdt,
            dm=dm,
            logger=mlf_logger,

            **args.attribution_kwargs
        )

        results_attribution = attributor.attribute()

    return results_train_best, results_val_best, results_test_best, results_attribution


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
        "test/mean/avg_score_pred_active/feature_importances",
        "test/mean/avg_score_pred_active/shap",

        "test/mean/avg_score_pred_inactive/feature_importances",
        "test/mean/avg_score_pred_inactive/shap",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "data_types": ["test"],
        "methods": [
            {"shap": {
                "postprocess": None
            }},
            {"feature_importances": {
                "postprocess": None
            }},
        ],
        "track_metrics": args.track_metrics,
        # "label": "active_g100",
        # "label_idx": 5,
        "label": "active_g10",
        "label_idx": 0,
        "references": Hergophores.ACTIVES_UNIQUE_,

        # "nr_samples": 100,
    }
    # args.attribution_kwargs = None

    # trainer/logging args
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.experiment_name = "herg_gbdt_best_kfold"
    args.run_name = "gbdt"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.seed = random.randint(0, 2 ** 32 - 1)
    args.patience = 10

    # data module args
    args.batch_size = 9999
    args.split_type = "random_kfold"
    args.split_size = (5, 0, 1)
    # args.split_type = "random"
    # args.split_size = (0.6, 0.2, 0.2)
    args.split_seed = random.randint(0, 2 ** 32 - 1)
    args.standardize = True

    # args.use_labels = ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"]
    args.use_labels = ["active_g10"]
    args.featurizer_name = "combined"
    args.featurizer_kwargs = {
        "fold": 1024,
        "radius": 3,
        "return_count": True,
        "use_chirality": True,
        "use_features": True,
    }
    # args.featurizer_n_jobs = 8

    # args.num_workers = multiprocessing.cpu_count()
    args.num_workers = 0
    args.cache_dir = "../../../" + "data/herg/"

    # model args
    args.gbdt_kwargs = {
        "subsample": 1.0,
        "n_estimators": 200,
        "learning_rate": 0.1,
        "gamma": 2.0,  # [0.0, 0.1, 0.2, 0.3, 0.4]
        "max_depth": 24,
        "min_child_weight": 1,
        "max_delta_step": 10,
        "colsample_bytree": 0.6,  # [0.3, 0.4, 0.5, 0.7]
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

    results = {}
    if "kfold" in args.split_type:
        kfold = Kfold(
            function=train_gbdt_kfold,
            args=args,
            **vars(args)
        )
        results = kfold.train()

    else:
        results_train_best, results_val_best, results_test_best, results_attribution = train_gbdt(args)
        results = results_test_best

        pprint(results_train_best)
        pprint(results_val_best)
        pprint(results_attribution)

    pprint(results)
