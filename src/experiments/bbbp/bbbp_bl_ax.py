import sys
from argparse import Namespace, ArgumentParser

from bbbp_bl import train_bl
from experiments import TuneAx


def train_evaluate(args: Namespace):

    # args.scheduler_params["decay_step"] = args.decay_step
    # args.scheduler_params["decay_rate"] = args.decay_rate

    args.hidden_size = [args.hidden_size] * args.nr_layers

    results_test, results_val_best, results_val_last, *_ = train_bl(args)

    metric = results_val_last[args.objective_name]
    # metric= random.random()

    return metric


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # ax args
    args.trials = 25
    args.trials_sobol = 10
    args.objective_name = "val/AUROC"
    args.minimize = False
    args.search_space = [
        # {"name": "batch_size", "type": "choice", "values": [128, 256, 512, 1024]},

        {"name": "nr_layers", "type": "range", "bounds": [3, 10]},
        {"name": "hidden_size", "type": "range", "bounds": [32, 256]},
        {"name": "dropout", "type": "choice", "values": [0.0, 0.05, 0.1, 0.3]},

        {"name": "lr", "type": "range", "bounds": [1e-5, 0.001], "log_scale": True},

        # {"name": "decay_step", "type": "choice", "values": [500, 2000, 8000]},
        # {"name": "decay_rate", "type": "choice", "values": [0.4, 0.8, 0.9, 0.95]},
    ]

    # trainer/logging args
    args.experiment_name = "bbbp_bl_4096_6_no_emb_ax_long1"
    args.tracking_uri = "https://mlflow.kriechbaumer.at"
    args.max_steps = 1000
    args.seed = 0
    args.patience = 50

    # data module args
    args.batch_size = 256
    args.split_seed = 0
    args.n_bits = 4096
    args.radius = 6
    args.chirality = True
    args.features = True

    args.num_workers = 8
    args.cache_dir = "../../../" + "data/molnet/bbbp/"

    # model args
    args.hidden_size = [256, 256, 256]

    args.virtual_batch_size = -1  # -1 do not use any batch normalization
    args.normalize_input = False

    args.lr = 0.001
    args.optimizer = "adam"
    # args.scheduler = "exponential_decay"
    # args.scheduler_params = {"decay_step": 50, "decay_rate": 0.95}

    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.0001}
    args.scheduler = "linear_with_warmup"
    args.scheduler_params = {"warmup_steps": 10}
    # args.scheduler_params={"warmup_steps": 0.01}

    # args.index_embeddings = True
    # args.categorical_indices = list(range(args.n_bits))
    # args.categorical_size = [2] * args.n_bits
    # args.embedding_dims = 1
    # args.embedding_dims = [1] * len(CovTypeDataModule.BINARY_COLUMNS)

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
