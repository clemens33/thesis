import json
import multiprocessing
import os
import sys
import tempfile
from argparse import Namespace, ArgumentParser
from pathlib import Path, PurePosixPath
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients
from mlflow.tracking import MlflowClient
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from baseline.rf import RandomForest
from datasets import HERGClassifierDataModule, Hergophores
from datasets.featurizer import calculate_ranking_scores
from tabnet_lightning import TabNetClassifier


class TabNetCaptum():
    def __init__(self, model: TabNetClassifier, target: int = 0, internal_batch_size: int = 128) -> None:
        super().__init__()

        self.model = model
        self.target = target
        self.internal_batch_size = internal_batch_size

    def integrated_gradients(self, data_loader, postprocess: str = "none", **kwargs) -> np.ndarray:
        def _forward(_inputs):
            _logits, _probs, *_ = self.model(_inputs)

            if _logits.ndim > 1:
                _logits = _logits[:, self.target]

            return _logits.reshape(len(_inputs), -1)

        ig = IntegratedGradients(_forward)

        inputs, _ = next(iter(data_loader))

        inputs = inputs.reshape(len(inputs), -1) if inputs.ndim == 1 else inputs
        inputs.requires_grad_(True)

        attributions = ig.attribute(inputs, target=self.target, internal_batch_size=len(inputs), **kwargs)
        attributions = attributions.detach().cpu().numpy()

        if postprocess == "normalize":
            attributions = _normalize(attributions)

        return attributions


class Attributor:
    """
    Simple wrapper class which handles the atomic attribution/weights calculation, atomic ranking and logging
    """

    def __init__(self,
                 model: Union[TabNetClassifier, RandomForest],
                 dm: HERGClassifierDataModule,
                 logger: Optional[MLFlowLogger] = None,
                 track_metrics: Optional[List[str]] = None,
                 types: Optional[List[str]] = None,
                 label: str = "active_g10",
                 label_idx: int = 0,
                 references: Union[List[str], List[Tuple[str, int]]] = None,
                 out_dir: str = str(tempfile.mkdtemp()) + "/",
                 out_fname: str = None,
                 nr_samples: Optional[int] = None,
                 threshold: Optional[float] = None,
                 model_attribution_kwargs: Optional[Dict] = None,
                 ):
        super(Attributor, self).__init__()

        self.types = ["test"] if types is None else types
        self.model = model
        self.dm = dm
        self.logger = logger
        self.track_metrics = track_metrics
        self.types = types
        self.label = label
        self.label_idx = label_idx
        self.references = [(rs, ra) for rs, ra in zip(*Hergophores.get())] if references is None else references
        self.out_dir = out_dir
        self.out_fname = out_fname
        self.nr_samples = nr_samples
        self.model_attribution_kwargs = model_attribution_kwargs if model_attribution_kwargs is not None else {}

        self.threshold = self._determine_treshold(.5) if threshold is None else threshold

    def _determine_treshold_tabnet(self, threshold_default: float = .5) -> float:
        metrics = Trainer().test(model=self.model, test_dataloaders=self.dm.train_dataloader())

        key = "test/threshold-t" + str(self.label_idx)
        if key in metrics[0]:
            return metrics[0][key]
        else:
            return threshold_default

    def _determine_treshold_rf(self, threshold_default: float = .5) -> float:
        metrics = self.model.test(self.dm.train_dataloader(), stage="test", log=False)

        key = "test/threshold-t" + str(self.label_idx)
        if key in metrics:
            return metrics[key]
        else:
            return threshold_default

    def _determine_treshold(self, threshold_default: float = .5) -> float:
        if isinstance(self.model, TabNetClassifier):
            return self._determine_treshold_tabnet(threshold_default)
        elif isinstance(self.model, RandomForest):
            self._determine_treshold_rf(threshold_default)
        else:
            return threshold_default

    def _attribution_tabnet(self, data_loader: DataLoader, type: str = "tabnet", **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """gets the tabnet attribution using the defined data_loader"""

        attribution, probs = [], []
        for batch in data_loader:
            _, _probs, _, _mask, *_ = self.model(*batch)

            attribution.append(_mask)
            probs.append(_probs)

        attribution = torch.cat(attribution).detach().cpu().numpy()
        probs = torch.cat(probs)
        preds = (probs > self.threshold).float()
        preds = preds.detach().cpu().numpy()

        preds = preds[:, self.label_idx] if preds.ndim > 1 else preds

        if type == "integrated_gradients":
            attribution = TabNetCaptum(self.model).integrated_gradients(data_loader, **kwargs)

        return attribution, preds, probs.detach().cpu().numpy()

    def _attribution_rf(self, data_loader: DataLoader, type: str = "global", postprocess: str = "none", **kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:

        preds, probs = self.model.predict(data_loader)

        if type == "treeinterpreter":
            attribution = self.model.contributions(data_loader)
            indices = np.expand_dims(preds, axis=(1, 2))
            attribution = np.take_along_axis(attribution, indices, axis=2).squeeze()
        elif type == "permutation":
            attribution = self.model.contributions_global_permutation(data_loader, **kwargs)
        elif type == "global":
            attribution = self.model.contributions_global(data_loader)
        else:
            raise ValueError(f"unknown type {type}")

        if postprocess == "normalize":
            attribution = _normalize(attribution)
        elif postprocess == "positive":
            attribution[attribution < .0] = .0
        elif postprocess == "relative":
            attribution[attribution > .0] += attribution[attribution > .0] * 2
            attribution[attribution < .0] *= -1

        pos_probs = probs[:, -1]

        return attribution, preds, pos_probs

    def _attribution(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(self.model, TabNetClassifier):
            return self._attribution_tabnet(data_loader, **self.model_attribution_kwargs)
        elif isinstance(self.model, RandomForest):
            return self._attribution_rf(data_loader, **self.model_attribution_kwargs)
        else:
            raise ValueError(f"model type is not supported {type(self.model)}")

    def _attribute(self, type: str) -> Tuple[Dict, List[Dict], str, str]:

        def _filter_mappings(featurizer_atomic_mappings, indices):
            featurizer_atomic_mappings_filtered = []
            for atomic_mappings in featurizer_atomic_mappings:
                featurizer_atomic_mappings_filtered.append([atomic_mappings[idx] for idx in indices])

            return featurizer_atomic_mappings_filtered

        featurizer_atomic_mappings = None
        if type == "train":
            data_loader = self.dm.train_dataloader()
            data = self.dm.data.iloc[self.dm.train_indices]

            if self.dm.featurizer_atomic_mappings is not None:
                featurizer_atomic_mappings = _filter_mappings(self.dm.featurizer_atomic_mappings, self.dm.train_indices)

        elif type == "val":
            data_loader = self.dm.val_dataloader()
            data = self.dm.data.iloc[self.dm.val_indices]

            if self.dm.featurizer_atomic_mappings is not None:
                featurizer_atomic_mappings = _filter_mappings(self.dm.featurizer_atomic_mappings, self.dm.val_indices)

        elif type == "test":
            data_loader = self.dm.test_dataloader()
            data = self.dm.data.iloc[self.dm.test_indices]

            if self.dm.featurizer_atomic_mappings is not None:
                featurizer_atomic_mappings = _filter_mappings(self.dm.featurizer_atomic_mappings, self.dm.test_indices)

        else:
            raise ValueError(f"unknown data type {type}")

        smiles = data["smiles"].tolist()
        labels = data[self.label].tolist()

        attribution, preds, _ = self._attribution(data_loader)
        labels = np.array(labels)

        if self.nr_samples:
            if featurizer_atomic_mappings is not None:
                for i in range(len(featurizer_atomic_mappings)):
                    featurizer_atomic_mappings[i] = featurizer_atomic_mappings[i][:self.nr_samples]

            smiles = smiles[:self.nr_samples]
            labels = labels[:self.nr_samples, ...]
            preds = preds[:self.nr_samples, ...]
            attribution = attribution[:self.nr_samples, ...]

        atomic_attributions = self.dm.atomic_attributions(
            smiles_or_mappings=smiles if featurizer_atomic_mappings is None else featurizer_atomic_mappings,
            feature_attributions=attribution)

        result, reference_results, df = calculate_ranking_scores(
            smiles=smiles,
            references=self.references,
            atomic_attributions=atomic_attributions,
            labels=labels,
            preds=preds,
        )

        out_fname = type + "_dataset" if not self.out_fname else self.out_fname
        out_name = self.out_dir + out_fname + "-" + "attribution_details-" + self.label

        out_path_df = out_name + ".tsv"
        df.to_csv(out_path_df, sep="\t")

        out_path_results = out_name + ".json"
        with open(out_path_results, "w") as f:
            results = [{"attribution_results": result}] + reference_results

            f.write(json.dumps(results))

        return result, reference_results, out_path_df, out_path_results

    def attribute(self, verbose: bool = False) -> Dict:

        value = lambda v: v if len(str(v)) <= 250 else "...value too long for mlflow - not inserted"

        metrics = {}
        for t in self.types:
            result, reference_results, out_path_df, out_path_results = self._attribute(type=t)

            if self.logger:
                self.logger.experiment.log_artifact(run_id=self.logger._run_id, local_path=out_path_results)

                if verbose:
                    self.logger.experiment.log_artifact(run_id=self.logger._run_id, local_path=out_path_df)

            for i, reference_result in enumerate(reference_results):
                reference_smile, reference_result_values = next(iter(reference_result.items()))

                if self.logger:
                    self.logger.experiment.log_param(run_id=self.logger._run_id, key="train/threshold-t" + str(self.label_idx),
                                                     value=value(self.threshold))
                    self.logger.experiment.log_param(run_id=self.logger._run_id, key="smile" + str(i), value=value(reference_smile))

                for k, v in reference_result_values.items():
                    key = t + "/" + "smile" + str(i) + "/" + k

                    if key in self.track_metrics and self.logger:
                        self.logger.experiment.log_metric(run_id=self.logger._run_id, key=key, value=v)

                    metrics[key] = v

            for k, v in result.items():
                key = t + "/" + k
                if key in self.track_metrics and self.logger:
                    self.logger.experiment.log_metric(run_id=self.logger._run_id, key=key, value=v)

                metrics[key] = v

        return metrics


def _normalize(data: np.ndarray) -> np.ndarray:
    _min = np.expand_dims(np.min(data, axis=1), axis=1)
    _max = np.expand_dims(np.max(data, axis=1), axis=1)

    data = (data - _min) / (_max - _min)

    return data


def attribution_fn(args: Namespace):
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
        split_seed=model.hparams.split_seed,
        split_type=args.split_type,
        split_size=args.split_size,
        use_labels=model.hparams.use_labels,
        featurizer_name=model.hparams.featurizer_name,
        featurizer_kwargs=model.hparams.featurizer_kwargs,
        featurizer_n_jobs=args.featurizer_n_jobs,
        featurizer_mp_context=args.featurizer_mp_context,
        featurizer_chunksize=args.featurizer_chunksize,
    )

    dm.prepare_data()
    dm.setup()

    attributor = Attributor(
        model=model,
        dm=dm,
        logger=mlf_logger,

        **args.attribution_kwargs
    )

    metrics = attributor.attribute()

    return metrics


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    args.track_metrics = []
    args.track_metrics += [
        "test/mean/avg_score_pred_active",
        "test/mean/avg_score_pred_inactive",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution params
    args.attribution_kwargs = {
        "types": ["test"],
        "track_metrics": args.track_metrics,
        # "label": "active_g100",
        # "label_idx": 5,
        "label": "active_g10",
        "label_idx": 0,
        "threshold": 0.5,
        "references": [(rs, ra) for rs, ra in zip(*Hergophores.get(Hergophores.ACTIVES_UNIQUE, by_activity=1))],
        "model_attribution_kwargs": {
            "type": "integrated_gradients",
        }
        # "nr_samples": 100,
    }
    # ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"]

    # logger/plot params
    args.experiment_name = "herg_tn_opt3"
    args.experiment_id = "177"
    # args.run_id = "0b563814b1c74588b4dbc4653d1f943b"
    args.run_id = "1c6e0a9892c4438e8772c61a4116ae5e"
    # args.run_id = "c244505250e24ba889c8a144488b926d"
    # args.run_id = "4e9cbbcdb36f4691a34af0ecfc1b94ca"
    # args.run_id = "b12fbb9514444737b9e37cab856514ec"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")

    p = list(Path(PurePosixPath("./" + args.experiment_id + "/" + args.run_id + "/checkpoints/")).glob("**/*.ckpt"))[0]
    args.checkpoint_name = p.name
    args.checkpoint_path = "./" + args.experiment_id + "/" + args.run_id + "/checkpoints/"

    args.max_steps = 1000000
    # args.max_steps = 100
    args.seed = 1234

    # data module args
    args.batch_size = 16384
    args.split_type = "random"
    args.split_size = (0.6, 0.2, 0.2)
    args.featurizer_n_jobs = 0
    args.featurizer_mp_context = "fork"
    args.featurizer_chunksize = 100

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

    metrics = attribution_fn(args)

    print(metrics)
