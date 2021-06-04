import copy
import sys
import uuid
from argparse import Namespace, ArgumentParser
from typing import Callable, Dict, List, Optional, Union

import mlflow
import numpy as np
from ax import Models, Experiment
from ax.core import TParameterization
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.service.managed_loop import optimize


class TuneAx:
    """
    Singleton class handling parameter optimization using ax platform's GPEI Gaussian Processes Expected Improvement implementation (https://ax.dev).
    Supports only single objective optimization at the moment
    """

    _instance = None

    def __init__(self,
                 function: Callable[[Namespace], float],
                 search_space: List[Dict],
                 objective_name: Optional[str] = None,
                 minimize: bool = False,
                 experiment_name: str = str(uuid.uuid4()),
                 trials: int = 20,
                 trials_sobol: int = 5,
                 seed: int = 0,
                 args: Optional[Namespace] = None,
                 tracking_uri: Optional[str] = None,
                 **kwargs) -> None:
        """

        Args:
            function (): function to optimize - accepts a single instance of Namespace as argument and must return a single float metric
            search_space (): defines the parameter search space
            objective_name (): objective name
            minimize (): is objective minimized or maximized
            experiment_name (): name of the experiment
            trials (): how many trials to run in total
            trials_sobol (): how many sobal samples to draw (quasi uniform samples) before starting GPEI
            seed (): seed used in ax
            args (): optional arguments which will be pass
            tracking_uri (): mlflow tracking uri
            **kwargs ():
        """
        super().__init__()

        assert trials_sobol <= trials, f"more sobol samples/trials {trials_sobol} defined than overall trials {trials}"

        if TuneAx._instance is not None:
            return TuneAx._instance

        self.function = function
        self.search_space = search_space
        self.objective_name = objective_name
        self.minimize = minimize
        self.trials = trials
        self.trials_sobol = trials_sobol
        self.seed = seed
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.args = args if args is not None else Namespace()

        self.generation_strategy = GenerationStrategy(name="Sobol+GPEI",
                                                      steps=[
                                                          GenerationStep(model=Models.SOBOL, num_trials=trials_sobol),
                                                          GenerationStep(model=Models.GPEI, num_trials=-1)]
                                                      )

        self.metrics = []

        # TODO add singleton gate to only allow single instance of TuneAx
        TuneAx._instance = self

    def optimize(self):

        best_parameters, values, experiment, model = optimize(
            parameters=self.search_space,
            total_trials=self.trials,
            random_seed=self.seed,
            evaluation_function=TuneAx.train_evaluate,
            objective_name=self.objective_name,
            minimize=self.minimize,
            generation_strategy=self.generation_strategy
        )

        self.log(best_parameters, experiment)

        return best_parameters, values

    def log(self, best_parameters: TParameterization, experiment: Experiment):
        """logging using mlflow - is done AFTER optimization is finished"""

        if self.tracking_uri is None:
            return

        def value(v):
            return v if len(str(v)) <= 250 else "...value too long for mlflow - not inserted"

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("objective_name", self.objective_name)
            mlflow.log_param("minimize", self.minimize)
            mlflow.log_param("trials", self.trials)
            mlflow.log_param("trials_sobol", self.trials_sobol)
            mlflow.log_param("seed", self.seed)
            # mlflow.log_param("tracking_uri", self.tracking_uri)

            for parameterization in self.search_space:
                mlflow.log_param("search_space/" + parameterization["name"], value(parameterization))

            for k, v in vars(self.args).items():
                mlflow.log_param("args/" + k, value(v))

            for k, v in best_parameters.items():
                mlflow.log_param("best/" + k, value(v))

            for trial in experiment.trials_expecting_data:
                arm = trial.arms[0]  # rewrite for multi arm support

                for k, v in arm.parameters.items():
                    mlflow.log_metric("param/" + k, value(v), step=trial.index)

                mlflow.log_metric(self.objective_name + "/objective_mean", trial.objective_mean, step=trial.index)

            for k, v in best_parameters.items():
                mlflow.log_metric("best/" + k, v)

            for i in range(len(self.metrics)):
                mlflow.log_metric(self.objective_name, self.metrics[i], step=i)

                if i > 0:
                    mlflow.log_metric(self.objective_name + "/mean_running", np.array(self.metrics[:i]).mean(), step=i)
                    mlflow.log_metric(self.objective_name + "/std_running", np.array(self.metrics[:i]).std(), step=i)

            mlflow.log_metric(self.objective_name + "/mean", np.array(self.metrics).mean(), step=trial.index)
            mlflow.log_metric(self.objective_name + "/std", np.array(self.metrics).std(), step=trial.index)

    @staticmethod
    def train_evaluate(parameterization: Dict):
        assert TuneAx._instance is not None, "TuneAx needs to be initialized first"

        args = copy.deepcopy(TuneAx._instance.args)
        function = TuneAx._instance.function

        # overwrite all arguments by the search space
        for key, value in parameterization.items():
            if key != "args":
                setattr(args, key, value)

        metric = function(args)

        TuneAx._instance.metrics.append(metric)

        return metric

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
#     args.experiment_name = "ax_pipeline_test2"
#     args.tracking_uri=os.getenv("TRACKING_URI", default="http://localhost:5000")
#     args.seed = 0  # model seed
#
#     # model args
#     args.x = 2
#     args.y = 3
#     args.bias = 2.2
#     args.dummy_long_param = "i am a long string" * 100
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
#     ax = TuneAx(
#         function=dummy_function,
#         args=args,
#
#         **vars(args)
#     )
#     ax.optimize()
