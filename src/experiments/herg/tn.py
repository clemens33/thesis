import multiprocessing
import os
import sys
import traceback
from argparse import Namespace, ArgumentParser
from typing import Dict, Tuple, List

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from datasets import HERGClassifierDataModule
from experiments.herg.attribution import Attributor
from experiments.kfold import Kfold
from tabnet_lightning import TabNetClassifier, TabNetTrainer


def train_tn_kfold(args: Namespace, split_size: Tuple[int, ...], split_seed: int) -> Dict[str, float]:
    """helper function to be called from Kfold implementation"""

    args.split_size = split_size
    args.split_seed = split_seed

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


def train_tn(args: Namespace, **kwargs) -> Tuple[Dict, Dict, Dict, Dict, TabNetClassifier, TabNetTrainer, HERGClassifierDataModule]:
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
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
        # use_labels=args.use_labels,
        use_cache=True,
        featurizer_name=args.featurizer_name,
        featurizer_kwargs=args.featurizer_kwargs
    )
    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    seed_everything(args.seed)

    # exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=args.seed)

    classifier = TabNetClassifier(
        input_size=dm.input_size,
        num_classes=dm.num_classes,
        ignore_index=HERGClassifierDataModule.IGNORE_INDEX,

        **vars(args),
    )

    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
                                    value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    callbacks = [
        ModelCheckpoint(
            monitor=args.objective_name,
            mode="max" if not args.minimize else "min",
        ),
        EarlyStopping(
            monitor=args.objective_name,
            patience=args.patience,
            mode="max" if not args.minimize else "min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if "callbacks" in kwargs:
        callbacks += kwargs["callbacks"]

    trainer = TabNetTrainer(
        gpus=1,
        terminate_on_nan=True,

        max_steps=args.max_steps,
        check_val_every_n_epoch=1,
        # terminate_on_nan=True,

        num_sanity_val_steps=-1,

        deterministic=True,
        # precision=16,

        callbacks=callbacks,
        logger=mlf_logger
    )
    trainer.log_hyperparameters(mlf_logger)

    try:
        trainer.fit(classifier, dm)
    except Exception as e:
        print(f"exception raised during training: {e}")

        print(traceback.format_exc())
        print(sys.exc_info())

    # gets the best validation metrics
    r = trainer.test(test_dataloaders=dm.val_dataloader())
    results_val_best = {}
    for k, v in r[0].items():
        results_val_best[k.replace("test", "val")] = v

    # gets the last validation metrics
    results_val_last = {}
    for k, v in trainer.callback_metrics.items():
        if "val" in k:
            results_val_last[k] = v.item() if isinstance(v, torch.Tensor) else v

    results_test = trainer.test(test_dataloaders=dm.test_dataloader())

    results_attribution = {}
    if args.attribution_kwargs is not None:
        model = TabNetClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        attributor = Attributor(
            model=model,
            dm=dm,
            logger=mlf_logger,

            **args.attribution_kwargs
        )

        results_attribution = attributor.attribute()

    return results_test[0], results_val_best, results_val_last, results_attribution, classifier, trainer, dm


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
        "test/mean/avg_score_true_active",
        "test/mean/avg_score_true_inactive",
    ]
    args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "types": ["test"],
        "track_metrics": args.track_metrics,
        # "label": "active_g100",
        # "label_idx": 5,
        "label": "active_g10",
        "label_idx": 0,
        # "nr_samples": 100,
    }

    # trainer/logging args
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.experiment_name = "herg_tn_kfold8"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 1000
    args.seed = 99
    args.patience = 100

    # data module args
    args.batch_size = 128
    args.split_type = "random_kfold"
    args.split_size = (5, 0, 1)
    # args.split_type = "random"
    # args.split_size = (0.6, 0.2, 0.2)
    args.split_seed = 99

    args.use_labels = ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"]

    args.featurizer_name = "combined"
    args.featurizer_kwargs = {
        "fold": 1024,
        "radius": 3,
        "return_count": True,
        "use_chirality": True,
        "use_features": True,
    }

    args.num_workers = 8  # multiprocessing.cpu_count()
    # args.num_workers = 0
    args.cache_dir = "../../../" + "data/herg/"

    # model args
    args.decision_size = 64
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 5
    args.gamma = 1.5

    args.lambda_sparse = 1e-6

    # args.virtual_batch_size = 32  # -1 do not use any batch normalization
    args.virtual_batch_size = -1  # -1 do not use any batch normalization
    # args.virtual_batch_size = 64
    # args.momentum = 0.1

    # args.normalize_input = True

    args.normalize_input = False
    # args.virtual_batch_size = 256  # -1 do not use any batch normalization

    args.lr = 0.0004
    # args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    args.optimizer = "adamw"
    args.optimizer_params = {"weight_decay": 0.0001}
    args.scheduler = "linear_with_warmup"
    # args.scheduler_params = {"warmup_steps": 10}
    args.scheduler_params = {"warmup_steps": 0.01}

    args.log_sparsity = True
    # args.log_sparsity = "verbose"
    # args.log_parameters = False
    args.attribution = ["test"]
    # args.attribution = None

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
            function=train_tn_kfold,
            args=args,
            **vars(args)
        )
        results = kfold.train()

    else:
        results, *_ = train_tn(args)

    print(results)
