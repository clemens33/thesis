import os
import sys
from argparse import Namespace, ArgumentParser
from typing import Dict, Tuple

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from baseline import MLPClassifier
from datasets import FirewallDataModule
from tabnet_lightning import TabNetTrainer


def train_bl(args: Namespace, **kwargs) -> Tuple[Dict, Dict, Dict, MLPClassifier, TabNetTrainer, FirewallDataModule]:
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri
    )

    dm = FirewallDataModule(
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        split_type="random",
        split_size=(0.8, 0.1, 0.1),
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    seed_everything(args.seed)

    exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=args.seed)

    classifier = MLPClassifier(
        input_size=dm.input_size,
        num_classes=len(FirewallDataModule.LABELS),

        **vars(args),
    )
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
                                    value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    callbacks = [
        ModelCheckpoint(
            monitor="val/Accuracy",
            mode="max",
        ),
        EarlyStopping(
            monitor="val/Accuracy",
            patience=args.patience,
            mode="max"
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if "callbacks" in kwargs:
        callbacks += kwargs["callbacks"]

    trainer = TabNetTrainer(
        gpus=1,

        max_steps=args.max_steps,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=-1,

        deterministic=True,
        # precision=16,

        callbacks=callbacks,
        logger=mlf_logger
    )
    trainer.log_hyperparameters(mlf_logger)

    trainer.fit(classifier, dm)

    r = trainer.test(test_dataloaders=dm.val_dataloader())
    results_val_best = {}
    for k, v in r[0].items():
        results_val_best[k.replace("test", "val")] = v

    results_val_last = {}
    for k, v in trainer.callback_metrics.items():
        if "val" in k:
            results_val_last[k] = v.item() if isinstance(v, torch.Tensor) else v

    results_test = trainer.test(test_dataloaders=dm.test_dataloader())

    return results_test[0], results_val_best, results_val_last, classifier, trainer, dm


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # trainer/logging args
    args.experiment_name = "fw_bl_m1"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 2000
    args.seed = 0  # model seed
    args.patience = 50

    # data module args
    args.batch_size = 256
    args.split_seed = 0
    args.num_workers = 0
    args.cache_dir = "../../../" + "data/firewall/"

    # model args
    args.hidden_size = [128] * 4
    args.dropout = 0.1
    args.normalize_input = True
    args.batch_norm = True

    args.lr = 1.000e-3
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.00005}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params={"warmup_steps": 0.01}

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

    results_val, results_test, *_ = train_bl(args)

    print(results_val)
