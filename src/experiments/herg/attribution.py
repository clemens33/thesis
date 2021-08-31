import json
import multiprocessing
import os
import sys
import tempfile
from argparse import Namespace, ArgumentParser
from typing import Optional, List, Dict

import torch
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger

from datasets import HERGClassifierDataModule, Hergophores
from datasets.featurizer import match
from tabnet_lightning import TabNetClassifier


def attribution(model: TabNetClassifier,
                dm: HERGClassifierDataModule,
                type: str,
                reference_smiles: Optional[List[str]] = None,
                out_dir: str = str(tempfile.mkdtemp()) + "/",
                out_fname: str = None,
                ):
    if type == "train":
        data_loader = dm.train_dataloader()
        smiles = dm.data.iloc[dm.train_indices]["smiles"]
    elif type == "val":
        data_loader = dm.val_dataloader()
        smiles = dm.data.iloc[dm.val_indices]["smiles"]
    elif type == "test":
        data_loader = dm.test_dataloader()
        smiles = dm.data.iloc[dm.test_indices]["smiles"]
    else:
        raise ValueError(f"unknown data type {type}")

    attribution, probs = [], []
    for batch in data_loader:
        _, _probs, _, _mask, *_ = model(*batch)

        attribution.append(_mask)
        probs.append(_probs)

    attribution = torch.cat(attribution)
    probs = torch.cat(probs)

    atomic_attributions = dm.atomic_attributions(smiles=smiles,
                                                 feature_attributions=attribution.detach().cpu().numpy())

    reference_smiles = Hergophores.get() if not reference_smiles else reference_smiles
    results, df = match(smiles=smiles, reference_smiles=reference_smiles, atomic_attributions=atomic_attributions)

    out_fname = type + "_dataset" if not out_fname else out_fname

    out_path_df = out_dir + out_fname + "-" + "attribution_details" + ".tsv"
    df.to_csv(out_path_df, sep="\t")

    out_path_results = out_dir + out_fname + "-" + "attribution_results" + ".json"
    with open(out_path_results, "w") as f:
        f.write(json.dumps(results))

    return results, out_path_df, out_path_results


def log_attribution(model: TabNetClassifier, dm: HERGClassifierDataModule, logger: MLFlowLogger, type: Optional[List[str]] = None) -> Dict:
    if type is None:
        type = ["train", "val", "test"]

    value = lambda v: v if len(str(v)) <= 250 else "...value too long for mlflow - not inserted"

    ret = {}
    for t in type:
        results, out_path_df, out_path_results = attribution(model, dm, type=t)

        logger.experiment.log_artifact(run_id=logger._run_id, local_path=out_path_df)
        logger.experiment.log_artifact(run_id=logger._run_id, local_path=out_path_results)

        for i, result in enumerate(results[:-1]):
            for smile, attribution_results in result.items():
                logger.experiment.log_param(run_id=logger._run_id, key="smile" + str(i), value=value(smile))
                logger.experiment.log_metric(run_id=logger._run_id, key=t + "/" + "smile" + str(i) + "-" + "mean_auroc",
                                             value=attribution_results["mean_auroc"])

                ret[t + "/" + "smile" + str(i) + "-" + "mean_auroc"] = attribution_results["mean_auroc"]

        attribution_summary = results[-1]["attribution_summary"]
        logger.experiment.log_metric(run_id=logger._run_id, key=t + "/" + "smile-mean_aurocs", value=attribution_summary["mean_aurocs"])

        ret[t + "/" + "smile-mean_aurocs"] = attribution_summary["mean_aurocs"]

    return ret


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

    results, out_path_df, out_path_results = attribution(model, dm, type="test")

    mlf_logger.experiment.log_artifact(run_id=mlf_logger._run_id, local_path=out_path_df)
    mlf_logger.experiment.log_artifact(run_id=mlf_logger._run_id, local_path=out_path_results)


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # logger/plot params
    args.experiment_name = "herg_tn_opt1"
    args.experiment_id = "152"
    args.run_id = "38a991230dca488b9b3d107e1ca9fcc7"
    # args.run_id = "c244505250e24ba889c8a144488b926d"
    # args.run_id = "4e9cbbcdb36f4691a34af0ecfc1b94ca"
    # args.run_id = "b12fbb9514444737b9e37cab856514ec"
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
