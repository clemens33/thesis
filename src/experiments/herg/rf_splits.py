import os
import sys
from argparse import Namespace, ArgumentParser
from pprint import pprint
from typing import Tuple, Dict

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import MLFlowLogger

from baseline.rf import RandomForest
from datasets import Hergophores, HERGClassifierDataModule
from experiments.grid_runner import GridRunner
from experiments.herg.attribution import Attribution
from experiments.herg.rf import train_rf_kfold, train_rf
from experiments.kfold import Kfold





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
            function=train_rf_kfold,
            args=args,
            **vars(args)
        )
        metrics = kfold.train()

    else:
        results_test, results_val_best, results_val_last, results_attribution, *_ = train_rf(args)

        metrics = {}
        for metric_name in args.track_metrics:
            if metric_name in results_test:
                metrics[metric_name] = results_test[metric_name]
            if metric_name in results_val_best:
                metrics[metric_name] = results_val_best[metric_name]
            elif metric_name in results_attribution:
                metrics[metric_name] = results_attribution[metric_name]

    return metrics


def train_rf(args: Namespace):
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

    rf = RandomForest(
        input_size=dm.input_size,
        num_classes=dm.num_classes,
        seed=args.seed,
        class_weight=class_weight,
        rf_kwargs=args.rf_kwargs,
        logger=mlf_logger,
    )
    results_train_best = rf.fit(dm.train_dataloader())
    results_val_best = rf.test(dm.val_dataloader(), stage="val")
    results_test_best = rf.test(dm.test_dataloader(), stage="test")

    results_attribution = {}
    if args.attribution_kwargs is not None:
        if "threshold" not in args.attribution_kwargs:
            key = "train/threshold-t" + str(args.attribution_kwargs["label_idx"])
            args.attribution_kwargs["threshold"] = results_train_best[key] if key in results_train_best else .5

        attributor = Attribution(
            model=rf,
            dm=dm,
            logger=mlf_logger,

            **args.attribution_kwargs
        )

        results_attribution = attributor.attribute()

    return results_train_best, results_val_best, results_test_best, results_attribution


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # metric tracking options
    args.track_metrics = [
        "val/AUROC",
        "val/Accuracy",
        "test/AUROC",
        "test/Accuracy",
    ]
    args.track_metrics += [
        "test/mean/avg_score_pred_active",
        "test/mean/avg_score_pred_inactive",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "data_types": ["test"],
        "track_metrics": args.track_metrics,
        # "label": "active_g100",
        # "label_idx": 5,
        "label": "active_g10",
        "label_idx": 0,
        "references": Hergophores.ACTIVES_UNIQUE_,
        "model_attribution_kwargs": {
            "type": "global",
            "n_repeats": 5,
        }
        # "nr_samples": 100,
    }
    # args.attribution_kwargs = None

    # [333, 7664, 9744, 1432, 1138 ]

    # trainer/logging args
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.experiment_name = "herg_rf_diff_seed1"
    args.run_name = "test1"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.seed = 1

    # data module args
    args.batch_size = 9999
    args.split_type = "random_kfold"
    args.split_size = (5, 0, 1)
    # args.split_type = "random"
    # args.split_size = (0.6, 0.2, 0.2)
    args.split_seed = 2

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
    args.rf_kwargs = {
        "n_estimators": 100,
        # "max_features": None
    }

    # args.log_sparsity = True
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
