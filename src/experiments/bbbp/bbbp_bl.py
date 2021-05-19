import sys
from argparse import Namespace, ArgumentParser
from typing import Dict, Tuple

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from baseline import MLPClassifier
from datasets import MolNetClassifierDataModule
from tabnet_lightning import TabNetTrainer


def train_bl(args: Namespace) -> Tuple[Dict, Dict, Dict, MLPClassifier, TabNetTrainer, MolNetClassifierDataModule]:
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri
    )

    dm = MolNetClassifierDataModule(
        name="bbbp",
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        split="random",
        split_size=(0.8, 0.1, 0.1),
        radius=args.radius,
        n_bits=args.n_bits,
        chirality=args.chirality,
        features=args.features,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        use_cache=True,
        noise_features=args.noise_features if hasattr(args, "noise_features") else None,
        noise=args.noise if hasattr(args, "noise") else None,
    )
    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    seed_everything(args.seed)

    exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=args.seed)

    classifier = MLPClassifier(
        input_size=dm.input_size,
        num_classes=len(dm.classes),
        class_weights=dm.class_weights,

        **vars(args),
    )
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
                                    value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    callbacks = [
        ModelCheckpoint(
            monitor="val/AUROC",
            mode="max",
        ),
        EarlyStopping(
            monitor="val/AUROC",
            patience=args.patience,
            mode="max"
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

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
    args.experiment_name = "bbbp_bl_random_features_12288"
    args.tracking_uri = "https://mlflow.kriechbaumer.at"
    args.max_steps = 1000
    args.seed = 0  # model seed
    args.patience = 50

    # data module args
    args.batch_size = 256
    args.split_seed = 0
    args.n_bits = 12288
    args.radius = 4
    args.chirality = True
    args.features = True
    args.noise_features = {
        "type": "replicate",
        "factor": 1.0,
        "position": "right",
    }

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.hidden_size = [211] * 5
    args.dropout = 0.1

    args.lr = 1.000e-5
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
