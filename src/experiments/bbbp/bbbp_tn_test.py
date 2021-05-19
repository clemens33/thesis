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
        print(f"start trial {t+1}/{args.trials}, test/AUROC_mean_running {np.array(aurocs_test).mean() if t > 1 else 0}")

        results_test, results_val_best, *_ = train_tn(args)

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
        for k, v in vars(args).items():
            mlflow.log_param(k, v)

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


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # test args
    args.trials = 20
    args.split_seed_init = 0

    # trainer/logging args
    args.experiment_name = "bbbp_tn_test_12288_4_no_emb_long2"
    args.tracking_uri = "https://mlflow.kriechbaumer.at"
    args.max_steps = 1000
    args.seed = 0  # model seed
    args.patience = 50

    # data module args
    args.batch_size = 256
    args.split_seed = 0
    args.n_bits = 12288
    args.radius = 4
    args.chirality = True
    args.features = True

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.decision_size = 64
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 10
    args.gamma = 1.0
    args.lambda_sparse = 0.01

    args.virtual_batch_size = -1  # -1 do not use any batch normalization
    args.normalize_input = False

    args.lr = 2.881e-4
    args.optimizer = "adam"
    #args.scheduler = "exponential_decay"
    #args.scheduler_params = {"decay_step": 800, "decay_rate": 0.377}

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
