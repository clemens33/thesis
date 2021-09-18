import multiprocessing
from typing import Dict, Union, List, Optional

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
import mlflow
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, AUROC

from tabnet_lightning.metrics import CustomAccuracy, CustomAUROC, postprocess_metric_output
from tabnet_lightning.utils import determine_objective


class RF:
    def __init__(self,
                 input_size: int,
                 num_classes: Union[int, List[int]],
                 ignore_index: int = -100,
                 class_weight: Union[Dict, List[Dict], str] = None,
                 seed: int = 0,
                 n_jobs: int = multiprocessing.cpu_count(),
                 verbose: int = 2,
                 pos_label: int = 1,
                 rf_kwargs=None,
                 **kwargs
                 ) -> None:
        super().__init__()

        if rf_kwargs is None:
            rf_kwargs = {}
        self.n_jobs = n_jobs
        self.seed = seed

        self.input_size = input_size
        self.ignore_index = ignore_index

        self.objective, self.num_classes, self.num_targets = determine_objective(num_classes)
        if self.objective not in ["binary", "binary-multi-target"]:
            raise NotImplementedError(f"{self.objective} not yet supported")

        self.pos_label = pos_label

        self.model = RandomForestClassifier(
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
            class_weight=class_weight, **rf_kwargs)

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.model.fit(X, y)

        return self._format_metrics(self.test(X, y), "train")

    def fit_(self, data_loader: DataLoader) -> Dict[str, float]:
        if len(data_loader) > 1:
            self.model.warm_start = True

            for X, y in data_loader:
                self.model.fit(X.cpu().numpy(), y.cpu().numpy())
                self.model.n_estimators += 1

            metrics = self.test_(data_loader)
        else:
            X, y = next(iter(data_loader))
            metrics = self.fit(X.cpu().numpy(), y.cpu().numpy())

        return metrics

    def predict(self, X: np.ndarray):
        y_hat = self.model.predict(X)
        probs = self.model.predict_proba(X)

        return y_hat, probs

    def test(self, X: np.ndarray, y: np.ndarray):
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
                idx = np.where(self.model.classes_ == self.pos_label)
                probs = probs[..., idx]
        else:
            raise NotImplementedError(f"{self.objective} not yet supported")

        self.metrics.update(torch.from_numpy(probs), torch.from_numpy(y))
        metrics = self.metrics.compute()

        return postprocess_metric_output(metrics)

    def test_(self, data_loader: DataLoader):
        self.metrics.reset()

        if len(data_loader) > 1:
            for X, y in data_loader:
                probs = self.model.predict_proba(X.cpu().numpy())

                self.metrics.update(torch.from_numpy(probs), torch.from_numpy(y))

            metrics = postprocess_metric_output(self.metrics.compute())
        else:
            X, y = next(iter(data_loader))
            metrics = self.test(X.cpu().numpy(), y.cpu().numpy())

        return metrics

    def _format_metrics(self, metrics: Dict, stage: str):
        _metrics = {}
        for k, v in metrics:
            key = stage + "/" + k

            _metrics[k] = v

        return _metrics

    # def log(self, metrics: Dict):
    #     """logging using mlflow - is done AFTER fit"""
    #
    #     if self.tracking_uri is None:
    #         return
    #
    #     def value(v):
    #         return v if len(str(v)) <= 250 else "...value too long for mlflow - not inserted"
    #
    #     mlflow.set_tracking_uri(self.tracking_uri)
    #     mlflow.set_experiment(self.experiment_name)
    #
    #     with mlflow.start_run():
    #         mlflow.log_param("experiment_name", self.experiment_name)
    #         mlflow.log_param("folds", self.nr_folds)
    #         mlflow.log_param("use_test_fold", self.use_test_fold)
    #         mlflow.log_param("track_metrics", value(self.track_metrics))
    #         # mlflow.log_param("tracking_uri", self.tracking_uri)
    #
    #         for k, v in vars(self.args).items():
    #             mlflow.log_param("args/" + k, value(v))
    #
    #         for metric_name in self.track_metrics:
    #             if metric_name in self.metrics:
    #                 for i in range(self.nr_folds):
    #                     mlflow.log_metric(metric_name, self.metrics[metric_name][i], step=i)
    #
    #                 mlflow.log_metric(metric_name + "/mean", np.array(self.metrics[metric_name]).mean())
    #                 mlflow.log_metric(metric_name + "/std", np.array(self.metrics[metric_name]).std())
