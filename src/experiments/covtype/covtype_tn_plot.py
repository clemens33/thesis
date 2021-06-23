import os
import sys
from argparse import Namespace, ArgumentParser

import mlflow
import torch
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger

from datasets import CovTypeDataModule
from tabnet_lightning import TabNetClassifier, TabNetTrainer


def plot_tn(args: Namespace):
    classifier = TabNetClassifier.load_from_checkpoint(args.checkpoint_path + args.checkpoint_name, strict=False)

    args = Namespace(**dict(classifier.hparams_initial, **vars(args)))

    _experiment = MlflowClient(args.tracking_uri).get_experiment_by_name(args.experiment_name)
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        artifact_location=_experiment.artifact_location if _experiment is not None else None
    )
    if getattr(args, "run_id", None) is not None:
        mlf_logger._run_id = args.run_id

    dm = CovTypeDataModule(
        batch_size=args.batch_size,
        num_workers=0,
        cache_dir=args.cache_dir,
        seed=args.seed,  # use same seed / random state as tabnet original implementation
    )

    dm.prepare_data()
    dm.setup()

    # dm.log_hyperparameters(mlf_logger)

    classifier.log_metrics = args.log_metrics
    classifier.log_sparsity = args.log_sparsity
    # mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
    #                                 value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    trainer = TabNetTrainer(gpus=1, logger=mlf_logger)

    classifier.log_masks = args.log_masks
    classifier.log_masks["feature_names"] = CovTypeDataModule.FEATURE_COLUMNS
    classifier.log_masks["out_fname"] = "masks_" + "validation_dataset_" + args.checkpoint_name.split(".")[0]

    classifier.log_rankings = args.log_rankings
    classifier.log_rankings["feature_names"] = CovTypeDataModule.FEATURE_COLUMNS
    classifier.log_rankings["out_fname"] = "ranks_" + "validation_dataset_" + args.checkpoint_name.split(".")[0]

    # gets the best validation metrics
    r = trainer.test(model=classifier, test_dataloaders=dm.val_dataloader())
    results_val_best = {}
    for k, v in r[0].items():
        results_val_best[k.replace("test", "val")] = v

    # gets the last validation metrics
    results_val_last = {}
    for k, v in trainer.callback_metrics.items():
        if "val" in k:
            results_val_last[k] = v.item() if isinstance(v, torch.Tensor) else v

    classifier.log_masks["out_fname"] = "masks_" + "test_dataset_" + args.checkpoint_name.split(".")[0]
    classifier.log_rankings["out_fname"] = "ranks_" + "test_dataset_" + args.checkpoint_name.split(".")[0]

    results_test = trainer.test(model=classifier, test_dataloaders=dm.test_dataloader())

    return results_test[0], results_val_best, results_val_last, classifier, trainer, dm


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # logger/plot params
    args.experiment_name = "covtype_tn1"
    args.experiment_id = "39"
    args.run_id = "3928ad8a8f084ba987d1aaa9f6d3704f"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.checkpoint_name = "epoch=2006-step=38132.ckpt"
    args.checkpoint_path = "./" + args.experiment_id + "/" + args.run_id + "/checkpoints/"

    args.max_steps = 1000000
    # args.max_steps = 100
    args.seed = 1234

    # data module args
    args.batch_size = 16384

    args.cache_dir = "../../../" + "data/uci/covtype/"

    args.log_masks = {
        "on_test_epoch_end": True,
        "nr_samples": 25,
        "normalize_inputs": True,
        "out_fname": "masks",
        "verbose": True,
    }

    args.log_rankings = {
        "on_test_epoch_end": True,
        "top_k": 100,
        "out_fname": "rankings",
    }

    args.log_metrics = False
    args.log_sparsity = False
    # args.log_parameters = True

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

    plot_tn(args)
