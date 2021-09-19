import multiprocessing
from argparse import Namespace
from typing import Dict, Union, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, AUROC
from treeinterpreter import treeinterpreter as ti

from tabnet_lightning.metrics import CustomAccuracy, CustomAUROC, postprocess_metric_output
from tabnet_lightning.utils import determine_objective


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

    def contributions(self, data: Union[DataLoader, np.ndarray]) -> np.ndarray:
        """
        only supports single target classifier

        using treeinterpreter - https://github.com/andosa/treeinterpreter

        additional links:
        - https://blog.datadive.net/random-forest-interpretation-conditional-feature-contributions/
        - https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e

        Args:
            data ():

        Returns:
            Contributions in the form of (n_samples, n_features)

        """
        if self.num_targets > 1:
            raise NotImplementedError(f"contribution for multi target classification not supported")

        if isinstance(data, DataLoader):
            if len(data) > 1:
                raise NotImplementedError("mini batches not supported yet")

            X, *_ = next(iter(data))
            X = X.cpu().numpy()
        else:
            X = data

        _, _, contributions = ti.predict(self.model, X, joint_contribution=False)
        return contributions

    def contributions_global_permutation(self, *data: Union[DataLoader, np.ndarray], n_repeats: int = 5) -> np.ndarray:
        """
        Calculates global contributions per feature using permutation

        Refer to https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html for details.

        Args:
            *data ():

        Returns:
            Contributions in the form of (n_samples, n_features) - but contributions are the same for each sample (global)

        """

        if isinstance(data[0], DataLoader):
            if len(data[0]) > 1:
                raise NotImplementedError("mini batches not supported yet")

            X, y = next(iter(data[0]))
            X = X.cpu().numpy()
            y = y.cpu().numpy()
        else:
            X, y = data

        print(f"start contributions_global_permutation ...")

        result = permutation_importance(
            self.model, X, y, n_repeats=n_repeats, random_state=self.seed, n_jobs=self.n_jobs)

        contributions = np.repeat(result.importances_mean[np.newaxis, :], len(X), axis=0)

        return contributions

    def contributions_global(self, *data: Union[DataLoader, np.ndarray]) -> np.ndarray:
        """
        Get global contributions per feature using impurity (default feature importance) - actually only considers the training data

        Refer to https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html for details.

        Args:
            *data (): - is only used to determine output size (n_sampleS)

        Returns:
            Contributions in the form of (n_samples, n_features) - but contributions are the same for each sample (global)

        """

        if isinstance(data[0], DataLoader):
            if len(data[0]) > 1:
                raise NotImplementedError("mini batches not supported yet")

            X, _ = next(iter(data[0]))
            X = X.cpu().numpy()
        else:
            X, _ = data

        contributions = self.model.feature_importances_

        contributions = np.repeat(contributions[np.newaxis, :], len(X), axis=0)

        return contributions

    def contributions_lime(self, data: Union[DataLoader, np.ndarray]):
        """

        using lime - https://github.com/marcotcr/lime

        Args:
            data ():

        Returns:

        """
        raise NotImplementedError(f"lime contribution not implemented yet")

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
