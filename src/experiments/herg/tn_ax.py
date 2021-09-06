import multiprocessing
import os
import sys
from argparse import Namespace, ArgumentParser

from datasets import Hergophores
from experiments import TuneAx
from experiments.kfold import Kfold
from tn import train_tn, train_tn_kfold


def train_evaluate(args: Namespace):
    args.feature_size = args.decision_size * 2

    args.optimizer_params["weight_decay"] = args.weight_decay
    # args.scheduler_params["decay_step"] = args.decay_step
    # args.scheduler_params["decay_rate"] = args.decay_rate

    if "kfold" in args.split_type:
        kfold = Kfold(
            function=train_tn_kfold,
            args=args,
            **vars(args)
        )
        results = kfold.train()

        metric = results[args.objective_name]
        # additional_metric = results.get("test/rs-mean_aurocs", .0)
        # additional_metric = results.get("val/Accuracy", .0)
        additional_metric = .0

        metric += additional_metric
    else:
        results_test, results_val_best, results_val_last, *_ = train_tn(args)

        metric = results_val_best[args.objective_name]

    return metric


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""
    # kfold options
    args.track_metrics = [
        "val/AUROC",
        "val/Accuracy",
        "test/AUROC",
        "test/Accuracy",
    ]
    args.track_metrics += [
        "test/mean/avg_score_label_active",
        "test/mean/avg_score_label_inactive",
        "test/mean/avg_score_true_active",
        "test/mean/avg_score_true_inactive",
    ]
    #args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    #args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "types": ["test"],
        "track_metrics": args.track_metrics,
        # "label": "active_g100",
        # "label_idx": 5,
        "label": "active_g10",
        "label_idx": 0,
        "references": [(rs, ra) for rs, ra in zip(*Hergophores.get(Hergophores.ACTIVES_UNIQUE, by_activity=1))]
        # "nr_samples": 100,
    }

    # ax args
    args.trials = 20
    args.trials_sobol = 5
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.search_space = [
        {"name": "batch_size", "type": "choice", "values": [32, 64, 128, 512]},

        {"name": "decision_size", "type": "choice", "values": [32, 64, 128, 256]},
        {"name": "nr_steps", "type": "range", "bounds": [3, 10]},
        {"name": "gamma", "type": "choice", "values": [1.0, 1.2, 1.5, 2.0]},

        # {"name": "virtual_batch_size", "type": "choice", "values": [-1, 8, 32, 64, 512]},
        # {"name": "momentum", "type": "choice", "values": [0.4, 0.3, 0.2, 0.1, 0.05, 0.02]},

        {"name": "lambda_sparse", "type": "choice", "values": [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1]},
        # {"name": "lr", "type": "choice", "values": [0.005, 0.01, 0.02, 0.025]},
        {"name": "lr", "type": "range", "bounds": [1e-4, 0.01], "log_scale": True},
        {"name": "weight_decay", "type": "choice", "values": [0.0, 1e-5, 1e-4, 0.001]},

        # {"name": "decay_step", "type": "choice", "values": [50, 200, 800]},
        # {"name": "decay_rate", "type": "choice", "values": [0.4, 0.8, 0.9, 0.95]},
        # {"name": "decay_rate", "type": "range", "bounds": [0.0, 1.0]},
    ]

    # trainer/logging args
    args.experiment_name = "herg_tn_ax8"
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 2000
    args.seed = 333
    args.patience = 1000

    # data module args
    args.batch_size = 128
    # args.split_type = "random_kfold"
    # args.split_size = (5, 0, 1)
    args.split_type = "random"
    args.split_size = (0.6, 0.2, 0.2)
    args.split_seed = 333
    args.use_labels = ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"]

    args.featurizer_name = "combined"  # ecfp + macc + tox
    args.featurizer_kwargs = {
        "fold": 1024,
        "radius": 3,
        "return_count": True,
        "use_chirality": True,
        "use_features": True,
    }

    args.num_workers = multiprocessing.cpu_count()
    # args.num_workers = 0

    args.cache_dir = "../../../" + "data/herg/"

    # model args
    args.decision_size = 64
    args.feature_size = args.decision_size * 2
    args.nr_layers = 2
    args.nr_shared_layers = 2
    args.nr_steps = 6
    args.gamma = 1.5

    # args.relaxation_type = "gamma_fixed"
    # args.alpha = 2.0
    # args.attentive_type = "sparsemax"

    # args.slope = 3.0
    # args.slope_type = "slope_fixed"

    # args.lambda_sparse = 1e-6
    # args.lambda_sparse = 0.1
    args.lambda_sparse = 0.001

    # args.virtual_batch_size = 32  # -1 do not use any batch normalization
    args.virtual_batch_size = -1  # -1 do not use any batch normalization
    # args.normalize_input = True

    args.normalize_input = False
    # args.virtual_batch_size = 256  # -1 do not use any batch normalization

    args.lr = 0.001
    # args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    args.optimizer = "adamw"
    args.optimizer_params = {"weight_decay": 0.0001}
    args.scheduler = "linear_with_warmup"
    # args.scheduler_params = {"warmup_steps": 10}
    args.scheduler_params = {"warmup_steps": 0.1}

    args.log_sparsity = True
    # args.log_sparsity = "verbose"
    # args.log_parameters = False
    args.attribution = ["test"]
    # args.attribution = None

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

    ax = TuneAx(
        function=train_evaluate,
        args=args,

        **vars(args)
    )
    best_parameters, values = ax.optimize()

    print(best_parameters)
    print(values)
