import sys
from argparse import Namespace, ArgumentParser
from typing import Dict, Tuple, List

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from datasets import MolNetClassifierDataModule
from experiments.models.index_emb_classifier import IndexEmbTabNetClassifier
from tabnet_lightning import TabNetClassifier, TabNetTrainer


def train_tn(args: Namespace, **kwargs) -> Tuple[Dict, Dict, Dict, TabNetClassifier, TabNetTrainer, MolNetClassifierDataModule]:
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        tags=args.tags if hasattr(args, "tags") else None
    )

    dm = MolNetClassifierDataModule(
        name=args.data_name,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        split_type="random",
        split_size=(0.8, 0.1, 0.1),
        radius=args.radius,
        n_bits=args.n_bits if hasattr(args, "n_bits") else None,
        chirality=args.chirality,
        features=args.features,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        use_cache=True,
        noise_features=args.noise_features if hasattr(args, "noise_features") else None,
        noise=args.noise if hasattr(args, "noise") else None,
        featurizer_name=args.featurizer_name
    )
    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    seed_everything(args.seed)

    # exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=args.seed)

    if getattr(args, "index_embeddings", False):
        classifier = IndexEmbTabNetClassifier(
            input_size=dm.input_size,
            num_classes=len(dm.classes),
            class_weights=dm.class_weights,

            **vars(args),
        )
    else:
        if getattr(args, "categorical_embeddings", False):
            args.categorical_indices = list(range(dm.input_size))
            args.categorical_size = dm.categorical_sizes.tolist()
            args.embedding_dims = 1
            # args.embedding_dims = [1] * args.n_bits

        classifier = TabNetClassifier(
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

    return results_test[0], results_val_best, results_val_last, classifier, trainer, dm


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # trainer/logging args
    args.experiment_name = "bbbp_tn_ecfc_norm_t1"
    args.tracking_uri = "https://mlflow.kriechbaumer.at"
    args.max_steps = 1000
    args.seed = 0
    args.patience = 50

    # data module args
    args.data_name = "bbbp"
    args.batch_size = 256
    args.split_seed = 0
    # args.n_bits = 12288
    args.radius = 6
    args.chirality = True
    args.features = True
    # args.noise_features = {
    #     "type": "zeros",
    #     "factor": 1.0,
    #     "position": "random",
    # }
    # args.noise = "zeros_standard_normal2"
    args.featurizer_name = "ecfc"

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.decision_size = 16
    args.feature_size = args.decision_size * 2
    args.nr_layers = 4
    args.nr_shared_layers = 0
    args.nr_steps = 8
    args.gamma = 1.2
    # args.lambda_sparse = 1e-6
    # args.lambda_sparse = 0.1
    args.lambda_sparse = 0.001

    args.virtual_batch_size = 32  # -1 do not use any batch normalization
    args.normalize_input = True

    # args.normalize_input = True
    # args.virtual_batch_size = 256  # -1 do not use any batch normalization

    args.lr = 0.01
    args.optimizer = "adam"
    args.scheduler = "exponential_decay"
    args.scheduler_params = {"decay_step": 100, "decay_rate": 0.95}

    # args.optimizer="adamw",
    # args.optimizer_params={"weight_decay": 0.0001}
    # args.scheduler = "linear_with_warmup"
    # args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params={"warmup_steps": 0.1}

    # args.index_embeddings = True
    args.categorical_embeddings = True
    # args.categorical_indices = list(range(args.n_bits))
    # args.categorical_size = [2] * args.n_bits
    # args.embedding_dims = 1
    # args.embedding_dims = [1] * args.n_bits

    # args.log_sparsity = "verbose"
    args.log_sparsity = True

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
