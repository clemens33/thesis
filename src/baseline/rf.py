import multiprocessing
from argparse import Namespace
from typing import Dict, Union, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, AUROC

from shared.metrics import CustomAccuracy, CustomAUROC, postprocess_metric_output
from shared.utils import determine_objective


class RandomForest:
    def __init__(self,
                 input_size: int,
                 num_classes: Union[int, List[int]],
                 ignore_index: int = -100,
                 class_weight: Union[Dict, List[Dict], str] = None,
                 seed: int = 0,
                 n_jobs: int = multiprocessing.cpu_count(),
                 verbose: int = 1,
                 pos_label: int = 1,
                 rf_kwargs: Optional[Dict] = None,
                 logger: Optional[MLFlowLogger] = None,
                 **kwargs
                 ) -> None:
        super().__init__()

        self.rf_kwargs = rf_kwargs if rf_kwargs is not None else {}
        self.n_jobs = n_jobs
        self.seed = seed

        self.input_size = input_size
        self.ignore_index = ignore_index

        self.objective, self.num_classes, self.num_targets = determine_objective(num_classes)
        if self.objective not in ["binary", "binary-multi-target"]:
            raise NotImplementedError(f"{self.objective} not yet supported")

        self.pos_label = pos_label
        self.logger = logger

        self.model = RandomForestClassifier(
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
            class_weight=class_weight, **self.rf_kwargs)

        self.metrics = MetricCollection(
            [
                CustomAccuracy(num_targets=self.num_targets, ignore_index=ignore_index),
                CustomAUROC(num_targets=self.num_targets, ignore_index=ignore_index, return_verbose=True),
            ]
            if self.objective in ["binary-multi-target", "binary"] else [
                Accuracy(),
                AUROC(num_classes=self.num_classes)
            ]
        )

    def fit(self, data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        if isinstance(data, DataLoader):
            if len(data) > 1:
                raise NotImplementedError("mini batches not supported yet")

            X, y = next(iter(data))
            X = X.cpu().numpy()
            y = y.cpu().numpy()
        else:
            X, y = data

        self._log_hyperparameters()

        self.model.fit(X, y)
        metrics = self.test(X, y, stage="train")

        self._log_metrics(metrics, finalize=True)

        return metrics

    def _test(self, X: np.ndarray, y: np.ndarray):
        self.metrics.reset()

        probs = self.model.predict_proba(X)

        if self.objective in ["binary-multi-target", "binary"]:

            if isinstance(probs, list):
                _probs = []
                for p, c in zip(probs, self.model.classes_):
                    idx = np.where(c == self.pos_label)[0].item()
                    _probs.append(p[..., idx])

                probs = np.stack(_probs, axis=1)
            else:
                idx = np.where(self.model.classes_ == self.pos_label)[0].item()
                probs = probs[..., idx]
        else:
            raise NotImplementedError(f"{self.objective} not yet supported")

        self.metrics.update(torch.from_numpy(probs), torch.from_numpy(y))
        metrics = self.metrics.compute()

        return metrics

    def test(self, *data: Union[DataLoader, np.ndarray], stage: str, log: bool = True) -> Dict[str, float]:
        if isinstance(data[0], DataLoader):
            if len(data[0]) > 1:
                raise NotImplementedError("mini batches not supported yet")

            X, y = next(iter(data[0]))
            X = X.cpu().numpy()
            y = y.cpu().numpy()
        else:
            if len(data) != 2:
                raise ValueError(f"X and y where not provided")

            X, y = data

        metrics = self._test(X, y)
        metrics = postprocess_metric_output(metrics, stage=stage)

        if log:
            self._log_metrics(metrics)

        return metrics

    def predict(self, data: Union[DataLoader, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, DataLoader):
            if len(data) > 1:
                raise NotImplementedError("mini batches not supported yet")

            X, *_ = next(iter(data))
            X = X.cpu().numpy()
        else:
            X = data

        y_hat = self.model.predict(X)
        probs = self.model.predict_proba(X)

        return y_hat, probs

    def _log_hyperparameters(self, ignore_param: List[str] = None, types: List = None):
        if self.logger:

            if types is None:
                types = [int, float, str, dict, list, bool, tuple]

            if ignore_param is None:
                ignore_param = ["model", "class_weight", "metrics", "logger"]

            params = {}
            for k, v in self.__dict__.items():
                if k not in ignore_param and not k.startswith("_"):
                    if type(v) in types:
                        params[k] = v

            params.update(self.model.get_params())

            params = Namespace(**params)

            self.logger.log_hyperparams(params)

    def _log_metrics(self, metrics: Dict, finalize: bool = False):
        if self.logger:
            self.logger.log_metrics(metrics)

            if finalize:
                self.logger.finalize()
