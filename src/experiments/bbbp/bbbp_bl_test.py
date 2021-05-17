import random
import sys
from argparse import Namespace, ArgumentParser
from typing import Dict

import mlflow
import numpy as np
from pytorch_lightning import seed_everything

from bbbp_tn import train_tn


def test(args: Namespace) -> Dict:
    args.split_seed = args.split_seed_init

    aurocs_test, aurocs_val, split_seeds = [], [], []
    for t in range(args.trials):
        results_val, results_test, *_ = train_tn(args)

        auroc_test = results_test[0]["test/AUROC"]
        auroc_val = results_val[0]["val/AUROC"]
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
        for k, v in vars(args).items():
            mlflow.log_param(k, v)

        for t in range(args.trials):
            mlflow.log_metric("split_seed", split_seeds[t], step=t)
            mlflow.log_metric("auroc_test", aurocs_test[t], step=t)
            mlflow.log_metric("auroc_test_mean_running", np.array(aurocs_test[:t]).mean(), step=t)
            mlflow.log_metric("auroc_test_std_running", np.array(aurocs_test[:t]).std(), step=t)

            mlflow.log_metric("auroc_val", aurocs_val[t], step=t)
            mlflow.log_metric("auroc_val_mean_running", np.array(aurocs_val[:t]).mean(), step=t)
            mlflow.log_metric("auroc_val_std_running", np.array(aurocs_val[:t]).std(), step=t)

        mlflow.log_metric("auroc_test_mean", auroc_test_mean)
        mlflow.log_metric("auroc_test_std", auroc_test_std)

        mlflow.log_metric("auroc_val_mean", auroc_val_mean)
        mlflow.log_metric("auroc_val_std", auroc_val_std)

    return {
        "auroc_test_mean": auroc_test_mean,
        "auroc_test_std": auroc_test_std,

        "auroc_val_mean": auroc_val_mean,
        "auroc_val_std": auroc_val_std,
    }


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # test args
    args.trials = 20
    args.split_seed_init = 0

    # trainer/logging args
    args.experiment_name = "bbbp_bl_test_12288"
    args.tracking_uri = "https://mlflow.kriechbaumer.at"
    args.max_steps = 1000
    args.seed = 0  # model seed
    args.patience = 20

    # data module args
    args.batch_size = 128
    args.split_seed = 0
    args.n_bits = 12288
    args.radius = 4
    args.chirality = True
    args.features = True

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.decision_size = 23
    args.feature_size = 23 * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 5
    args.gamma = 1.2
    args.lambda_sparse = 0.001

    args.virtual_batch_size = -1  # -1 do not use any batch normalization
    args.normalize_input = False

    args.lr = 0.0003525269090350661
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.00005}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params={"warmup_steps": 0.01}

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

    results = test(args)

    print(results)
