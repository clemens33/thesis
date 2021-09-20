import itertools
import json
import multiprocessing
import os
import sys
import tempfile
from argparse import Namespace, ArgumentParser
from pathlib import Path, PurePosixPath
from timeit import default_timer as timer
from typing import Optional, List, Dict, Union, Tuple

import captum.attr as ca
import numpy as np
import torch
from mlflow.tracking import MlflowClient
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch.utils.data import DataLoader

from baseline.rf import RandomForest
from datasets import HERGClassifierDataModule, Hergophores
from datasets.featurizer import calculate_ranking_scores
from tabnet_lightning import TabNetClassifier


class TabNetCaptum:
    def __init__(self, model: TabNetClassifier, target: int = 0, device: str = "cuda") -> None:
        super().__init__()

        self.model = model
        self.target = target
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        class _Model(nn.Module):
            def __init__(_self, model, target):
                super().__init__()

                _self.model = model
                _self.target = target

            def forward(_self, _inputs):
                return self._forward(_inputs)

        self._model = _Model(self.model, self.target)

    def _forward(self, _inputs):
        """forward wrapper - takes care of target handling"""
        _logits, *_ = self.model(_inputs)

        if _logits.ndim > 1:
            _logits = _logits[:, self.target]

        return _logits.reshape(len(_inputs), -1)

    def _inputs(self, data_loader: DataLoader) -> torch.Tensor:
        """prepare inputs"""
        inputs, _ = next(iter(data_loader))

        inputs = inputs.reshape(len(inputs), -1) if inputs.ndim == 1 else inputs
        inputs.requires_grad_(True)

        return inputs.to(self.device)

    def integrated_gradients(self, data_loader, **method_kwargs) -> np.ndarray:
        method = ca.IntegratedGradients(self._forward)

        inputs = self._inputs(data_loader)

        start = timer()
        print("integrated gradients start")
        attributions = method.attribute(inputs, target=0, internal_batch_size=len(inputs), **method_kwargs)
        print(f"integrated gradients runtime: {timer() - start} sec")

        attributions = attributions.detach().cpu().numpy()

        return attributions

    def noise_tunnel_ig(self, data_loader, **method_kwargs) -> np.ndarray:
        """noise tunnel for integrated gradients - https://captum.ai/api/noise_tunnel.html"""

        ig = ca.IntegratedGradients(self._forward)
        method = ca.NoiseTunnel(ig)

        inputs = self._inputs(data_loader)

        start = timer()
        print("noise tunnel integrated gradients start")
        attributions = method.attribute(inputs, target=0, internal_batch_size=len(inputs), **method_kwargs)
        print(f"noise tunnel integrated gradients runtime: {timer() - start} sec")

        attributions = attributions.detach().cpu().numpy()

        return attributions

    def saliency(self, data_loader, **method_kwargs) -> np.ndarray:
        """saliency - https://captum.ai/docs/algorithms#saliency"""

        method = ca.Saliency(self._forward)

        start = timer()
        print("saliency start")
        attributions = method.attribute(self._inputs(data_loader), target=0, **method_kwargs)
        print(f"saliency runtime: {timer() - start} sec")

        attributions = attributions.detach().cpu().numpy()

        return attributions

    def input_x_gradient(self, data_loader, **method_kwargs) -> np.ndarray:
        """input times gradient - https://captum.ai/docs/algorithms#input_x_gradient"""

        method = ca.InputXGradient(self._forward)

        start = timer()
        print("input_x_gradient start")
        attributions = method.attribute(self._inputs(data_loader), target=0, **method_kwargs)
        print(f"input_x_gradient runtime: {timer() - start} sec")

        attributions = attributions.detach().cpu().numpy()

        return attributions

    def occlusion(self, data_loader, **method_kwargs) -> np.ndarray:
        """occlusion - https://captum.ai/docs/algorithms#occlusion"""

        method = ca.Occlusion(self._forward)

        start = timer()
        print("occlusion start")
        attributions = method.attribute(self._inputs(data_loader), target=0, **method_kwargs)
        print(f"occlusion runtime: {timer() - start} sec")

        attributions = attributions.detach().cpu().numpy()

        return attributions

    def shapley_value_sampling(self, data_loader, **method_kwargs) -> np.ndarray:
        """shapley value sampling - https://captum.ai/docs/algorithms#shapley_value_sampling"""

        method = ca.ShapleyValueSampling(self._forward)

        start = timer()
        print("shapley_value_sampling start")
        attributions = method.attribute(self._inputs(data_loader), target=0, **method_kwargs)
        print(f"shapley_value_sampling runtime: {timer() - start} sec")

        attributions = attributions.detach().cpu().numpy()

        return attributions

    def permutation(self, data_loader, **method_kwargs) -> np.ndarray:
        """occlusion - https://captum.ai/docs/algorithms#shapley_value_sampling"""

        method = ca.FeaturePermutation(self._forward)

        start = timer()
        print("permutation start")
        attributions = method.attribute(self._inputs(data_loader), target=0, **method_kwargs)
        print(f"permutation runtime: {timer() - start} sec")

        attributions = attributions.detach().cpu().numpy()

        return attributions

    # def lrp(self, data_loader, **method_kwargs) -> np.ndarray:
    #     """lrp - https://captum.ai/docs/algorithms#lrp - not working identity is not supported by captum"""
    #
    #     method = ca.LRP(self._model)
    #
    #     start = timer()
    #     print("lrp start")
    #     attributions = method.attribute(self._inputs(data_loader), target=0, **method_kwargs)
    #     print(f"lrp runtime: {timer() - start} sec")
    #
    #     attributions = attributions.detach().cpu().numpy()
    #
    #     return attributions

    # def deep_lift(self, data_loader, multiply_by_inputs: bool = True, **method_kwargs) -> np.ndarray:
    #     """
    #     deep lift - https://captum.ai/api/deep_lift.html
    #
    #     not working atm due to captum limitation with parameter sharing networks
    #
    #     Args:
    #         data_loader ():
    #         multiply_by_inputs ():
    #         **method_kwargs ():
    #
    #     Returns:
    #
    #     """
    #
    #     method = ca.DeepLift(self._model, multiply_by_inputs=multiply_by_inputs)
    #
    #     start = timer()
    #     print("deep lift start")
    #     attributions = method.attribute(self._inputs(data_loader), target=self.target, **method_kwargs)
    #     print(f"deep lift runtime: {timer() - start} sec")
    #
    #     attributions = attributions.detach().cpu().numpy()
    #
    #     return attributions


class Attributor:
    """
    Simple wrapper class which handles the atomic attribution/weights calculation, atomic ranking and logging
    """

    def __init__(self,
                 model: Union[TabNetClassifier, RandomForest],
                 dm: HERGClassifierDataModule,
                 methods: List[Dict],
                 logger: Optional[MLFlowLogger] = None,
                 track_metrics: Optional[List[str]] = None,
                 data_types: Optional[List[str]] = None,
                 label: str = "active_g10",
                 label_idx: int = 0,
                 references: Union[List[str], List[Tuple[str, int]]] = None,
                 out_dir: str = str(tempfile.mkdtemp()) + "/",
                 out_fname: str = None,
                 nr_samples: Optional[int] = None,
                 threshold: Optional[float] = None,
                 device: str = "cuda",
                 ):
        super(Attributor, self).__init__()

        self.data_types = ["test"] if data_types is None else data_types
        self.methods = methods

        self.dm = dm
        self.logger = logger
        self.track_metrics = track_metrics

        self.label = label
        self.label_idx = label_idx
        self.references = [(rs, ra) for rs, ra in zip(*Hergophores.get())] if references is None else references
        self.out_dir = out_dir
        self.out_fname = out_fname
        self.nr_samples = nr_samples

        self.model = model
        self.device = torch.device(device)
        self.model.eval()
        self.model.to(self.device)

        self.threshold = self._determine_threshold(.5) if threshold is None else threshold

    def _determine_threshold_tabnet(self, threshold_default: float = .5) -> float:
        metrics = Trainer().test(model=self.model, test_dataloaders=self.dm.train_dataloader())

        key = "test/threshold-t" + str(self.label_idx)
        if key in metrics[0]:
            return metrics[0][key]
        else:
            return threshold_default

    def _determine_threshold_rf(self, threshold_default: float = .5) -> float:
        metrics = self.model.test(self.dm.train_dataloader(), stage="test", log=False)

        key = "test/threshold-t" + str(self.label_idx)
        if key in metrics:
            return metrics[key]
        else:
            return threshold_default

    def _determine_threshold(self, threshold_default: float = .5) -> float:
        if isinstance(self.model, TabNetClassifier):
            return self._determine_threshold_tabnet(threshold_default)
        elif isinstance(self.model, RandomForest):
            self._determine_threshold_rf(threshold_default)
        else:
            return threshold_default

    def _attribution_tabnet(self, data_loader: DataLoader, method: str = "default", **method_kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """gets the tabnet attribution using the defined data_loader"""

        attribution, probs = [], []
        for inputs, labels in data_loader:
            _, _probs, _, _mask, *_ = self.model(inputs.to(self.device))

            attribution.append(_mask)
            probs.append(_probs)

        attribution = torch.cat(attribution).detach().cpu().numpy()
        probs = torch.cat(probs)
        preds = (probs > self.threshold).float()
        preds = preds.detach().cpu().numpy()

        preds = preds[:, self.label_idx] if preds.ndim > 1 else preds

        postprocess = method_kwargs.pop("postprocess", None)

        if method == "default":
            # default refers to tabnet mask
            attributions = attribution
        elif method == "integrated_gradients":
            attributions = TabNetCaptum(self.model).integrated_gradients(data_loader, **method_kwargs)
        elif method == "noise_tunnel_ig":
            attributions = TabNetCaptum(self.model).noise_tunnel_ig(data_loader, **method_kwargs)
        elif method == "saliency":
            attributions = TabNetCaptum(self.model).saliency(data_loader, **method_kwargs)
        elif method == "input_x_gradient":
            attributions = TabNetCaptum(self.model).input_x_gradient(data_loader, **method_kwargs)
        elif method == "occlusion":
            attributions = TabNetCaptum(self.model).occlusion(data_loader, **method_kwargs)
        elif method == "shapley_value_sampling":
            attributions = TabNetCaptum(self.model).shapley_value_sampling(data_loader, **method_kwargs)
        elif method == "permutation":
            attributions = TabNetCaptum(self.model).permutation(data_loader, **method_kwargs)
        else:
            raise ValueError(f"unknown attribution method {method}")

        attributions = _postprocess(attributions, postprocess)

        return attributions, preds, probs.detach().cpu().numpy()

    def _attribution_rf(self, data_loader: DataLoader, method: str = "default", **method_kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:

        preds, probs = self.model.predict(data_loader)

        postprocess = method_kwargs.pop("postprocess", None)

        if method == "default":
            attributions = self.model.contributions_global(data_loader)
        elif method == "treeinterpreter":
            attributions = self.model.contributions(data_loader)
            indices = np.expand_dims(preds, axis=(1, 2))
            attributions = np.take_along_axis(attributions, indices, axis=2).squeeze()
        elif method == "permutation":
            attributions = self.model.contributions_global_permutation(data_loader, **method_kwargs)
        else:
            raise ValueError(f"unknown type {method}")

        attributions = _postprocess(attributions, postprocess)

        pos_probs = probs[:, -1]

        return attributions, preds, pos_probs

    def _attribution(self, data_loader: DataLoader, method: str, **method_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(self.model, TabNetClassifier):
            return self._attribution_tabnet(data_loader, method=method, **method_kwargs)
        elif isinstance(self.model, RandomForest):
            return self._attribution_rf(data_loader, method=method, **method_kwargs)
        else:
            raise ValueError(f"model type is not supported {type(self.model)}")

    def _attribute(self, data_type: str, method: str, **method_kwargs) -> Tuple[Dict, List[Dict], str, str]:

        def _filter_mappings(featurizer_atomic_mappings, indices):
            featurizer_atomic_mappings_filtered = []
            for atomic_mappings in featurizer_atomic_mappings:
                featurizer_atomic_mappings_filtered.append([atomic_mappings[idx] for idx in indices])

            return featurizer_atomic_mappings_filtered

        featurizer_atomic_mappings = None
        if data_type == "train":
            data_loader = self.dm.train_dataloader()
            data = self.dm.data.iloc[self.dm.train_indices]

            if self.dm.featurizer_atomic_mappings is not None:
                featurizer_atomic_mappings = _filter_mappings(self.dm.featurizer_atomic_mappings, self.dm.train_indices)

        elif data_type == "val":
            data_loader = self.dm.val_dataloader()
            data = self.dm.data.iloc[self.dm.val_indices]

            if self.dm.featurizer_atomic_mappings is not None:
                featurizer_atomic_mappings = _filter_mappings(self.dm.featurizer_atomic_mappings, self.dm.val_indices)

        elif data_type == "test":
            data_loader = self.dm.test_dataloader()
            data = self.dm.data.iloc[self.dm.test_indices]

            if self.dm.featurizer_atomic_mappings is not None:
                featurizer_atomic_mappings = _filter_mappings(self.dm.featurizer_atomic_mappings, self.dm.test_indices)

        else:
            raise ValueError(f"unknown data type {data_type}")

        smiles = data["smiles"].tolist()
        labels = data[self.label].tolist()

        attribution, preds, _ = self._attribution(data_loader, method=method, **method_kwargs)
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

        out_fname = method + "-" + data_type + "_dataset" if not self.out_fname else self.out_fname
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
        for t, m in itertools.product(self.data_types, self.methods):
            method_name = next(iter(m))
            method_kwargs = m[method_name] if m[method_name] is not None else {}

            result, reference_results, out_path_df, out_path_results = self._attribute(data_type=t, method=method_name, **method_kwargs)

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
                    key = key + "/" + method_name if method_name != "default" else key

                    if key in self.track_metrics and self.logger:
                        self.logger.experiment.log_metric(run_id=self.logger._run_id, key=key, value=v)

                    metrics[key] = v

            for k, v in result.items():
                key = t + "/" + k
                key = key + "/" + method_name if method_name != "default" else key

                if key in self.track_metrics and self.logger:
                    self.logger.experiment.log_metric(run_id=self.logger._run_id, key=key, value=v)

                metrics[key] = v

        return metrics


def _normalize(data: np.ndarray) -> np.ndarray:
    _min = np.expand_dims(np.min(data, axis=1), axis=1)
    _max = np.expand_dims(np.max(data, axis=1), axis=1)

    data = (data - _min) / (_max - _min)

    return data


def _postprocess(attributions: np.ndarray, postprocess: Optional[str] = None) -> np.ndarray:
    if postprocess == "normalize":
        attributions = _normalize(attributions)
    elif postprocess == "positive":
        attributions[attributions < .0] = .0
    elif postprocess == "relative":
        attributions[attributions > .0] += attributions[attributions > .0] * 2
        attributions[attributions < .0] *= -1

    return attributions


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
        # "test/mean/avg_score_pred_active",
        "test/mean/avg_score_pred_inactive",
        "integrated_gradients/test/mean/avg_score_pred_inactive",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution params
    args.attribution_kwargs = {
        "data_types": ["test"],
        "methods": [
            {"default": {
                "postprocess": None
            }},
            {"integrated_gradients": {
                "postprocess": None
            }},
            {"saliency": {
                "postprocess": None,
                "abs": False,  # Returns absolute value of gradients if set to True
            }},
            {"input_x_gradient": {
                "postprocess": None
            }},
            # {"occlusion": {
            #     "sliding_window_shapes": (1,),
            #     "perturbations_per_eval": 1,
            #     "show_progress": True,
            #     "postprocess": None
            # }},
            # {"shapley_value_sampling": {
            #     "n_samples": 10,  # The number of feature permutations tested
            #     "perturbations_per_eval": 1,
            #     "show_progress": True,  # takes around 30-40 min for default args
            #     "postprocess": None
            # }},
            # {"permutation": {
            #     "perturbations_per_eval": 1,
            #     "show_progress": True,  # takes around 30-40 min for default args
            #     "postprocess": None
            # }},
            # {"noise_tunnel_ig": {
            #     "postprocess": None
            # }}
        ],
        "track_metrics": args.track_metrics,
        "label": "active_g10",
        "label_idx": 0,
        "threshold": 0.5,
        "references": Hergophores.ACTIVES_UNIQUE_,

        # "nr_samples": 100,
    }
    # ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"]

    # logger/plot params
    args.experiment_name = "herg_tn_attr1"
    args.experiment_id = "190"
    args.run_id = "f2abb80eda84417593a67a535402eb72"

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
