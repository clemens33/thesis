import multiprocessing
import os
import sys
from argparse import Namespace, ArgumentParser
from typing import List

import numpy as np
import torch
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger

from datasets import HERGClassifierDataModule
from datasets.featurizer import match
from tabnet_lightning import TabNetClassifier, TabNetTrainer


def match_fn(args: Namespace):
    model = TabNetClassifier.load_from_checkpoint(args.checkpoint_path + args.checkpoint_name, strict=False)

    args = Namespace(**dict(model.hparams_initial, **vars(args)))

    _experiment = MlflowClient(args.tracking_uri).get_experiment_by_name(args.experiment_name)
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        artifact_location=_experiment.artifact_location if _experiment is not None else None
    )
    if getattr(args, "run_id", None) is not None:
        mlf_logger._run_id = args.run_id

    dm = HERGClassifierDataModule(
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count(),
        cache_dir=args.cache_dir,
        split_seed=model.hparams.split_seed,  # use same seed / random state as tabnet original implementation
        split_type=args.split_type,
        split_size=args.split_size,
        featurizer_name=model.hparams.featurizer_name,
        featurizer_kwargs=model.hparams.featurizer_kwargs,
    )

    dm.prepare_data()
    dm.setup()

    # dm.log_hyperparameters(mlf_logger)

    model.log_metrics = args.log_metrics
    model.log_sparsity = args.log_sparsity
    # mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
    #                                 value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    attribution, probs = [], []
    for batch in dm.train_dataloader():
        _, _probs, _, _mask, *_ = model(*batch)

        attribution.append(_mask)
        probs.append(_probs)

    attribution = torch.cat(attribution)
    probs = torch.cat(probs)

    smiles = dm.data.iloc[dm.train_indices]
    smiles = smiles["smiles"].tolist()

    # feature_attributions = attribution.detach().cpu().numpy()

    hergo = "c1ccccc1Cc1ccccc1"
    atomic_attributions = dm.atomic_attributions(smiles=smiles,
                                                 feature_attributions=attribution.detach().cpu().numpy())  # , postprocess="normalize")

    aurocs = match(smiles=smiles, reference_smile=hergo, atomic_attributions=atomic_attributions)

    print(np.nanmax(aurocs))
    print(np.nanmean(aurocs))
    print(sum(~np.isnan(aurocs)))


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # logger/plot params
    args.experiment_name = "herg_tn_opt1"
    args.experiment_id = "152"
    args.run_id = "38a991230dca488b9b3d107e1ca9fcc7"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.checkpoint_name = "epoch=5-step=221.ckpt"
    args.checkpoint_path = "./" + args.experiment_id + "/" + args.run_id + "/checkpoints/"

    args.max_steps = 1000000
    # args.max_steps = 100
    args.seed = 1234

    # data module args
    args.batch_size = 16384
    args.split_type = "random"
    args.split_size = (0.6, 0.2, 0.2)

    args.cache_dir = "../../../" + "data/herg/"

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

    match_fn(args)
