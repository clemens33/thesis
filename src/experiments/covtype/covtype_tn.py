import sys
from argparse import Namespace, ArgumentParser
from typing import Dict, Tuple, List

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from datasets import CovTypeDataModule
from tabnet_lightning import TabNetClassifier, TabNetTrainer


def train_tn(args: Namespace) -> Tuple[List[Dict], List[Dict], TabNetClassifier, TabNetTrainer, CovTypeDataModule]:
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri
    )

    dm = CovTypeDataModule(
        batch_size=args.batch_size,
        num_workers=8,
        cache_dir=args.cache_dir,
        seed=args.seed,  # use same seed / random state as tabnet original implementation
    )

    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    seed_everything(args.seed)

    exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=args.seed)

    if args.num_embeddings > 0:
        args.categorical_indices = list(range(len(CovTypeDataModule.NUMERICAL_COLUMNS), CovTypeDataModule.NUM_FEATURES, 1))
        args.categorical_size = [args.num_embeddings] * len(CovTypeDataModule.BINARY_COLUMNS)
        args.embedding_dims = 1
        #args.embedding_dims = [1] * len(CovTypeDataModule.BINARY_COLUMNS)

    classifier = TabNetClassifier(
        input_size=CovTypeDataModule.NUM_FEATURES,
        num_classes=CovTypeDataModule.NUM_LABELS,

        **vars(args),
    )
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
                                    value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    callbacks = [
        ModelCheckpoint(
            monitor="val/Accuracy",
            mode="max",
        ),
        # EarlyStopping(
        #     monitor="val/AUROC",
        #     patience=args.patience,
        #     mode="max"
        # ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = TabNetTrainer(
        gpus=1,

        max_steps=args.max_steps,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=-1,

        deterministic=True,
        # precision=16,

        gradient_clip_algorithm="value",
        gradient_clip_val=2000,

        callbacks=callbacks,
        logger=mlf_logger
    )
    trainer.log_hyperparameters(mlf_logger)

    trainer.fit(classifier, dm)

    results_test = trainer.test(test_dataloaders=dm.test_dataloader())
    results_val = trainer.validate(val_dataloaders=dm.val_dataloader())

    return results_val, results_test, classifier, trainer, dm


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # trainer/logging args
    args.experiment_name = "covtype_tn1"
    args.tracking_uri = "https://mlflow.kriechbaumer.at"
    args.max_steps = 1000000
    #args.max_steps = 100
    args.seed = 0

    # data module args
    args.batch_size = 16384

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/uci/covtype/"

    # model args
    args.decision_size = 64
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 5
    args.gamma = 1.5
    args.lambda_sparse = 0.0001

    args.virtual_batch_size = 512
    args.momentum = 0.3
    args.normalize_input = True

    args.lr = 0.02
    args.optimizer = "adam"
    args.scheduler = "exponential_decay"
    args.scheduler_params = {"decay_step": 500, "decay_rate": 0.95}

    args.num_embeddings = 2

    # args.optimizer="adamw",
    # args.optimizer_params={"weight_decay": 0.0001},
    # args.scheduler="linear_with_warmup",
    # args.scheduler_params={"warmup_steps": 0.1},
    # args.scheduler_params={"warmup_steps": 10},

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

    results_val, results_test, *_ = train_tn(args)

    print(results_val)
