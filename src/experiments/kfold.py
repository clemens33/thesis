import uuid
from argparse import Namespace
from typing import Callable, Optional, Tuple, List, Dict

import mlflow
import numpy as np


class Kfold:
    """simple wrapper class for kfold training - includes metric tracking and logging using mlflow"""

    def __init__(self,
                 function: Callable[[Namespace, Tuple[int, ...], int], Dict[str, float]],
                 track_metrics: List[str],
                 experiment_name: str = str(uuid.uuid4()),
                 nr_folds: int = 5,
                 seed: int = 0,
                 use_test_fold: bool = True,
                 args: Optional[Namespace] = None,
                 tracking_uri: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__()

        self.function = function

        self.track_metrics = track_metrics
        self.experiment_name = experiment_name
        self.nr_folds = nr_folds
        self.seed = seed
        self.use_test_fold = use_test_fold

        self.tracking_uri = tracking_uri
        self.run_name = "summary"
        self.args = args if args is not None else Namespace()

        self.metrics = {}

    def train(self) -> Dict[str, float]:
        """
        Calls the provided function n (fold) times - keeps track of defined metrics

        Returns:
            Dictionary containing the mean of the individually tracked metrics

        """

        for f in range(self.nr_folds):
            val_fold = f
            test_fold = f + 1 if f + 1 < self.nr_folds else 0

            if self.use_test_fold:
                split_size = (self.nr_folds, val_fold, test_fold)
            else:
                split_size = (self.nr_folds, val_fold)

            metrics = self.function(self.args, split_size, self.seed)

            for metric_name in self.track_metrics:
                if metric_name in metrics:
                    values = self.metrics.get(metric_name, [])
                    values.append(metrics[metric_name])

                    self.metrics[metric_name] = values

        self.log()

        metrics = {}
        for metric_name in self.track_metrics:
            if metric_name in self.metrics:
                metrics[metric_name] = float(np.array(self.metrics[metric_name]).mean())

        return metrics

    def log(self):
        """logging using mlflow - is done AFTER all folds are trained"""

        if self.tracking_uri is None:
            return

        def value(v):
            return v if len(str(v)) <= 250 else "...value too long for mlflow - not inserted"

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("folds", self.nr_folds)
            mlflow.log_param("use_test_fold", self.use_test_fold)
            mlflow.log_param("track_metrics", value(self.track_metrics))
            # mlflow.log_param("tracking_uri", self.tracking_uri)

            for k, v in vars(self.args).items():
                mlflow.log_param("args/" + k, value(v))

            for metric_name in self.track_metrics:
                if metric_name in self.metrics:
                    for i in range(self.nr_folds):
                        mlflow.log_metric(metric_name, self.metrics[metric_name][i], step=i)

                    mlflow.log_metric(metric_name + "/mean", np.array(self.metrics[metric_name]).mean())
                    mlflow.log_metric(metric_name + "/std", np.array(self.metrics[metric_name]).std())
