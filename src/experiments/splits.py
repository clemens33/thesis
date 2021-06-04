import random
from argparse import Namespace
from typing import Dict, Callable

import mlflow
import numpy as np
from pytorch_lightning import seed_everything


def splits(function: Callable, args: Namespace) -> Dict:
    args.split_seed = args.split_seed_init

    aurocs_test, aurocs_val, split_seeds = [], [], []
    for t in range(args.trials):
        print(f"{args.experiment_name} - start trial {t+1}/{args.trials}, test/AUROC_mean_running {np.array(aurocs_test).mean() if t > 1 else 0}")

        results_test, results_val_best, *_ = function(args)

        auroc_test = results_test["test/AUROC"]
        auroc_val = results_val_best["val/AUROC"]
        # auroc_test = random.random()
        # auroc_val = random.random()

        split_seeds.append(args.split_seed)
        aurocs_test.append(auroc_test)
        aurocs_val.append(auroc_val)

        seed_everything(args.split_seed)
        args.split_seed = random.randint(0, 2 ** 32 - 1)

    #
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    auroc_test_mean = np.array(aurocs_test).mean()
    auroc_test_std = np.array(aurocs_test).std()

    auroc_val_mean = np.array(aurocs_val).mean()
    auroc_val_std = np.array(aurocs_val).std()

    with mlflow.start_run():

        def value(v):
            return v if len(str(v)) <= 250 else "...value too long for mlflow - not inserted"

        for k, v in vars(args).items():
            mlflow.log_param(k, value(v))

        for t in range(args.trials):
            mlflow.log_metric("trail/split_seed", split_seeds[t], step=t)
            mlflow.log_metric("trial/test/AUROC", aurocs_test[t], step=t)
            mlflow.log_metric("trial/test/AUROC_mean_running", np.array(aurocs_test[:t]).mean(), step=t)
            mlflow.log_metric("trial/test/AUROC_std_running", np.array(aurocs_test[:t]).std(), step=t)

            mlflow.log_metric("trial/val/AUROC", aurocs_val[t], step=t)
            mlflow.log_metric("trial/val/AUROC_mean_running", np.array(aurocs_val[:t]).mean(), step=t)
            mlflow.log_metric("trial/val/AUROC_std_running", np.array(aurocs_val[:t]).std(), step=t)

        mlflow.log_metric("trial/test/AUROC_mean", auroc_test_mean)
        mlflow.log_metric("trial/test/AUROC_std", auroc_test_std)

        mlflow.log_metric("trial/val/AUROC_mean", auroc_val_mean)
        mlflow.log_metric("trial/val/AUROC_std", auroc_val_std)

    return {
        "auroc_test_mean": auroc_test_mean,
        "auroc_test_std": auroc_test_std,

        "auroc_val_mean": auroc_val_mean,
        "auroc_val_std": auroc_val_std,
    }