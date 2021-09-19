import copy
import itertools
import uuid
from argparse import Namespace
from typing import Callable, Dict, Optional, Any, List

import mlflow
import numpy as np


class GridRunner:

    def __init__(self,
                 function: Callable[[Namespace], Any],
                 grid_space: Dict,
                 track_metrics: List[str],
                 experiment_name: str = str(uuid.uuid4()),
                 max_trials: Optional[int] = None,
                 args: Optional[Namespace] = None,
                 tracking_uri: Optional[str] = None,
                 **kwargs) -> None:
        """

        Args:
            function (): function to optimize - accepts a single instance of Namespace as argument and must return a single float metric
            grid_space (): defines the parameter search space
            objective_name (): objective name
            experiment_name (): name of the experiment
            max_trials ()
            args (): optional arguments which will be pass to function
            tracking_uri (): mlflow tracking uri
            **kwargs ():
        """
        super().__init__()

        self.function = function

        for k, v in grid_space.items():
            if not isinstance(v, list):
                grid_space[k] = [v]
        self.grid_space = grid_space

        self.track_metrics = track_metrics
        self.max_trials = max_trials if max_trials is not None else 2 ** 32 - 1
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.args = args if args is not None else Namespace()

        self.metrics = {}

    def __iter__(self):
        param_keys = self.grid_space.keys()
        param_values = self.grid_space.values()

        for instance in itertools.product(*param_values):
            yield dict(zip(param_keys, instance))

    def __len__(self):
        return len(list(itertools.product(*self.grid_space.values())))

    def run(self) -> Dict[str, float]:
        for trial, parameterization in enumerate(iter(self)):
            print(f"start trial {trial + 1}/{len(self)} - parameterization: {parameterization}")

            if trial > self.max_trials:
                print(f"stopping grid search - max trials {self.max_trials} reached!")

            args = copy.deepcopy(self.args)
            # overwrite all arguments by the search space
            for key, value in parameterization.items():
                if key != "args":
                    setattr(args, key, value)

            metrics = self.function(args)

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
        """logging using mlflow - is done AFTER all runs are finished"""

        if self.tracking_uri is None:
            return

        def value(v):
            return v if len(str(v)) <= 250 else "...value too long for mlflow - not inserted"

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("max_trials", self.max_trials)
            # mlflow.log_param("seed", self.seed)
            # mlflow.log_param("tracking_uri", self.tracking_uri)

            for k, v in self.grid_space.items():
                mlflow.log_param("grid_space/" + k, value(v))

            for k, v in vars(self.args).items():
                mlflow.log_param("args/" + k, value(v))

            for metric_name in self.track_metrics:
                if metric_name in self.metrics:
                    for i in range(len(self.metrics[metric_name])):
                        mlflow.log_metric(metric_name, self.metrics[metric_name][i], step=i)

                    mlflow.log_metric(metric_name + "/mean", np.array(self.metrics[metric_name]).mean())
                    mlflow.log_metric(metric_name + "/std", np.array(self.metrics[metric_name]).std())
