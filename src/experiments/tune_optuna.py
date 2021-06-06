import copy
import sys
import uuid
from argparse import Namespace, ArgumentParser
from typing import Callable, Dict, List, Optional, Union

import mlflow
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import RandomSampler, TPESampler
from optuna.pruners import NopPruner, MedianPruner


class TuneOptuna:
    """
    Singleton class handling parameter optimization using Optuna - supports only single objective optimization
    """

    _instance = None

    def __init__(self,
                 function: Callable[[Namespace], float],
                 search_space: List[Dict],
                 objective_name: Optional[str] = None,
                 minimize: bool = False,
                 experiment_name: str = str(uuid.uuid4()),
                 trials: int = 20,
                 seed: int = 0,
                 args: Optional[Namespace] = None,
                 tracking_uri: Optional[str] = None,
                 sampler_name: str = "tpe",
                 pruner_name: Optional[str] = None,
                 pruner_warmup_steps: int = 10,
                 **kwargs) -> None:
        """

        Args:
            function: function to optimize - accepts a single instance of Namespace as argument and must return a single float metric
            search_space: defines the parameter search space using the ax format
            objective_name: objective name
            minimize: is objective minimized or maximized
            experiment_name: name of the experiment
            trials: how many trials to run in total
            seed: seed used in ax
            args: optional arguments which will be passed to function
            tracking_uri: mlflow tracking uri
            sampler_name: name of the optuna sampler used - defaults to tpe
            pruner_name: name of the optuna pruner used - defaults to None
            **kwargs:
        """
        super().__init__()

        if TuneOptuna._instance is not None:
            return TuneOptuna._instance

        self.function = function
        self.search_space = search_space
        self.objective_name = objective_name
        self.minimize = minimize
        self.trials = trials

        self.seed = seed
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.args = args if args is not None else Namespace()

        self.sampler_name = sampler_name
        self.pruner_name = pruner_name

        if sampler_name == "tpe":
            self.sampler = TPESampler(seed=seed)
        elif sampler_name == "random":
            self.sampler = RandomSampler(seed=seed)
        else:
            raise ValueError(f"sampler {sampler_name} not available")

        if pruner_name == "median":
            self.pruner = MedianPruner(n_warmup_steps=pruner_warmup_steps)
        elif pruner_name is None:
            self.pruner = NopPruner()
        else:
            raise ValueError(f"pruner {pruner_name} not available")

        self.study = optuna.create_study(
            sampler=self.sampler,
            pruner=self.pruner,
            direction="minimize" if minimize else "maximize",
            study_name=experiment_name
        )

        self.metrics = []

        # TODO add singleton gate to only allow single instance of TuneOptuna
        TuneOptuna._instance = self

    def optimize(self):

        self.study.optimize(
            func=TuneOptuna.objective,
            n_trials=self.trials
        )

        best_trial = self.study.best_trial

        self.log()

        return best_trial.params, best_trial.value

    def log(self):
        """logging using mlflow - is done AFTER optimization is finished"""

        if self.tracking_uri is None:
            return

        def value(v):
            return v if len(str(v)) <= 250 else "...value too long for mlflow - not inserted"

        best_trial = self.study.best_trial

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("objective_name", self.objective_name)
            mlflow.log_param("minimize", self.minimize)
            mlflow.log_param("trials", self.trials)
            mlflow.log_param("seed", self.seed)
            # mlflow.log_param("tracking_uri", self.tracking_uri)

            for parameterization in self.search_space:
                mlflow.log_param("search_space/" + parameterization["name"], value(parameterization))

            for k, v in vars(self.args).items():
                mlflow.log_param("args/" + k, value(v))

            for k, v in best_trial.params.items():
                mlflow.log_param("best/" + k, value(v))

            # log all trails params + value
            for trial in self.study.trials:
                for k, v in trial.params.items():
                    mlflow.log_metric("param/" + k, value(v), step=trial.number)

                mlflow.log_metric(self.objective_name + "/value", trial.value, step=trial.number)

            # log best params + value metric
            for k, v in best_trial.params.items():
                mlflow.log_metric("best/" + k, v, step=best_trial.number)
            mlflow.log_metric("best/" + self.objective_name, best_trial.value, step=best_trial.number)

            # log metric progression and mean
            for i in range(len(self.metrics)):
                mlflow.log_metric(self.objective_name, self.metrics[i], step=i)

                if i > 0:
                    mlflow.log_metric(self.objective_name + "/mean_running", np.array(self.metrics[:i]).mean(), step=i)
                    mlflow.log_metric(self.objective_name + "/std_running", np.array(self.metrics[:i]).std(), step=i)

            mlflow.log_metric(self.objective_name + "/mean", np.array(self.metrics).mean(), step=trial.number)
            mlflow.log_metric(self.objective_name + "/std", np.array(self.metrics).std(), step=trial.number)

    @staticmethod
    def parameterization(trial: optuna.trial.Trial, search_space: List[Dict], args: Namespace) -> Namespace:
        """parameterization function using the ax search space format"""

        def _type(values: List) -> type:
            return type(values[0])

        for s in search_space:
            name = s["name"]

            if s["type"] == "choice":
                value = trial.suggest_categorical(name, choices=s["values"])
            elif s["type"] == "range":
                log_scale = s["log_scale"] if hasattr(s, "log_scale") else False

                low = s["bounds"][0]
                high = s["bounds"][-1]

                if _type(s["bounds"]) == float:
                    value = trial.suggest_float(name, low, high, log=log_scale)
                elif _type(s["bounds"]) == int:
                    value = trial.suggest_int(name, low, high, log=log_scale)
                else:
                    raise ValueError(f"value type {_type(s['bounds'])} is unknown")
            else:
                raise ValueError(f"search space type {s['type']} is unknown")

            setattr(args, name, value)

        return args

    @staticmethod
    def objective(trial: optuna.trial.Trial):
        assert TuneOptuna._instance is not None, "TuneOptuna needs to be initialized first"

        # get activate instance
        instance = TuneOptuna._instance
        args = copy.deepcopy(instance.args)
        function = instance.function

        args = TuneOptuna.parameterization(trial, instance.search_space, args)

        kwargs = {}
        if instance.pruner_name:
            callbacks = getattr(args, "callbacks", [])

            callbacks += [
                PyTorchLightningPruningCallback(trial, monitor=instance.objective_name)
            ]

            kwargs["callbacks"] = callbacks
            # setattr(args, "callbacks", callbacks)

        metric = function(args, **kwargs)

        instance.metrics.append(metric)

        best_trial = np.array(instance.metrics).argmin() if instance.minimize else np.array(instance.metrics).argmax()
        best_metric = np.array(instance.metrics).min() if instance.minimize else np.array(instance.metrics).max()

        # print(f"{instance.experiment_name} - {len(instance.metrics)}/{instance.trials} - parameters: {parameterization}")
        print(f"{instance.experiment_name} - {len(instance.metrics)}/{instance.trials} - {instance.objective_name}: {metric}")
        print(f"{instance.experiment_name} - trail {best_trial} is best with {instance.objective_name}: {best_metric}")

        return metric

#
# # test -> TODO move to pytest
# def manual_args(args: Namespace) -> Namespace:
#     """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""
#
#     # ax args
#     args.trials = 20
#     args.search_space = [
#         {"name": "x", "type": "range", "bounds": [0.0, 2.0]},
#         {"name": "y", "type": "range", "bounds": [0.0, 2.0]},
#     ]
#     args.objective_name = "z"
#     args.minimize = True
#
#     # trainer/logging args
#     args.experiment_name = "optuna_pipeline_test1"
#     args.tracking_uri=os.getenv("TRACKING_URI", default="http://localhost:5000")
#     args.seed = 0  # model seed
#
#     # model args
#     args.x = -1
#     args.y = -1
#     args.bias = 2.2
#     args.dummy_long_param = "i am a long string" * 2
#
#     return args
#
#
# def run_cli() -> Namespace:
#     parser = ArgumentParser()
#
#     # parser.add_argument("--test", type=bool, default=True)
#     args = parser.parse_args()
#
#     # if no arguments have been provided we use manually set arguments - for debugging/dev
#     args = manual_args(args) if len(sys.argv) <= 1 else args
#
#     return args
#
#
# def dummy_function(args: Namespace):
#     x = args.x
#     y = args.y
#     bias = args.bias
#
#     z = x ** 2 + y ** 2 + bias
#
#     return z
#
#
# if __name__ == "__main__":
#     args = run_cli()
#
#     opt = TuneOptuna(
#         function=dummy_function,
#         args=args,
#         sampler_name="random",
#
#         **vars(args)
#     )
#     results = opt.optimize()
#     print(results)
