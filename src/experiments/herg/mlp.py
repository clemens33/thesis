import os
import random
import sys
import traceback
from argparse import Namespace, ArgumentParser
from typing import Dict, Tuple

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from datasets import HERGClassifierDataModule, Hergophores
from experiments.herg.attribution import Attribution
from experiments.kfold import Kfold
from shared.trainer import CustomTrainer
from baseline.classifier import MLPClassifier


def train_mlp_kfold(args: Namespace, split_size: Tuple[int, ...], split_seed: int) -> Dict[str, float]:
    """helper function to be called from Kfold implementation"""

    args.split_size = split_size
    args.split_seed = split_seed

    results_test, results_val_best, results_val_last, results_attribution, *_ = train_mlp(args)

    metrics = {}
    for metric_name in args.track_metrics:
        if metric_name in results_test:
            metrics[metric_name] = results_test[metric_name]
        if metric_name in results_val_best:
            metrics[metric_name] = results_val_best[metric_name]
        elif metric_name in results_attribution:
            metrics[metric_name] = results_attribution[metric_name]

    return metrics


def train_mlp(args: Namespace, **kwargs) -> Tuple[Dict, Dict, Dict, Dict, Dict, MLPClassifier, CustomTrainer, HERGClassifierDataModule]:
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=args.run_name if hasattr(args, "run_name") else None,
        tracking_uri=args.tracking_uri,
        tags=args.tags if hasattr(args, "tags") else None
    )

    dm = HERGClassifierDataModule(
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        split_type=args.split_type,
        split_size=args.split_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        use_labels=args.use_labels,
        use_cache=True,
        featurizer_name=args.featurizer_name,
        featurizer_kwargs=args.featurizer_kwargs,
        featurizer_mp_context="fork",
        featurizer_chunksize=100,
        featurizer_n_jobs=args.featurizer_n_jobs if hasattr(args, "featurizer_n_jobs") else 0,
        standardize=args.standardize if hasattr(args, "standardize") else True,
    )
    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    seed_everything(args.seed)

    # exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=args.seed)

    classifier = MLPClassifier(
        input_size=dm.input_size,
        num_classes=dm.num_classes,
        ignore_index=HERGClassifierDataModule.IGNORE_INDEX,

        **vars(args),
    )

    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
                                    value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    callbacks = [
        ModelCheckpoint(
            monitor=args.checkpoint_objective if hasattr(args, "checkpoint_objective") else "val/loss",
            mode="max" if not args.checkpoint_minimize else "min",
        ),
        EarlyStopping(
            monitor=args.patience_objective if hasattr(args, "patience_objective") else "val/loss",
            patience=args.patience,
            mode="min" if args.patience_minimize else "max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if "callbacks" in kwargs:
        callbacks += kwargs["callbacks"]

    max_epochs = (args.max_steps // len(dm.train_dataloader())) + 1

    trainer = CustomTrainer(
        gpus=1,
        terminate_on_nan=True,

        max_steps=args.max_steps,
        max_epochs=max_epochs,
        # check_val_every_n_epoch=1,
        val_check_interval=0.5,

        # terminate_on_nan=True,

        num_sanity_val_steps=-1,
        stochastic_weight_avg=args.stochastic_weight_avg if hasattr(args, "stochastic_weight_avg") else False,

        deterministic=True,
        gradient_clip_val=args.gradient_clip_val if hasattr(args, "gradient_clip_val") else 0,
        # precision=16,

        callbacks=callbacks,
        logger=mlf_logger
    )
    trainer.log_hyperparameters(mlf_logger)

    try:
        trainer.fit(classifier, dm)
    except Exception as e:
        print(f"exception raised during training: {e}")

        print(traceback.format_exc())
        print(sys.exc_info())

    # gets the best training metrics
    r = trainer.test(test_dataloaders=dm.train_dataloader())
    results_train_best = {}
    for k, v in r[0].items():
        results_train_best[k.replace("test", "train")] = v

    # gets the best validation metrics
    r = trainer.test(test_dataloaders=dm.val_dataloader())
    results_val_best = {}
    for k, v in r[0].items():
        results_val_best[k.replace("test", "val")] = v

    # gets the last validation metrics
    results_val_last = {}
    for k, v in trainer.callback_metrics.items():
        if "val" in k:
            results_val_last[k] = v.item() if isinstance(v, torch.Tensor) else v

    results_test = trainer.test(test_dataloaders=dm.test_dataloader())

    results_attribution = {}
    if args.attribution_kwargs is not None:
        model = MLPClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        if "threshold" not in args.attribution_kwargs:
            key = "train/threshold-t" + str(args.attribution_kwargs["label_idx"])
            args.attribution_kwargs["threshold"] = results_train_best[key] if key in results_train_best else .5

        attributor = Attribution(
            model=model,
            dm=dm,
            logger=mlf_logger,

            **args.attribution_kwargs
        )

        results_attribution = attributor.attribute()

    return results_test[0], results_val_best, results_val_last, results_attribution, results_train_best, classifier, trainer, dm


def manual_args(args: Namespace) -> Namespace:
    """function only called if no arguments have been passed to the script - mostly used for dev/debugging"""

    # metrics + options
    args.track_metrics = [
        "val/loss",
        "val/AUROC",
        "val/Accuracy",
        "val/sparsity_mask",
        "test/loss",
        "test/AUROC",
        "test/Accuracy",
        "test/sparsity_mask",
    ]
    args.track_metrics += [
        # "test/mean/avg_score_pred_active",
        "test/mean/avg_score_pred_inactive",
        "test/mean/avg_score_pred_inactive/tabnet",
        "test/mean/avg_score_pred_inactive/integrated_gradients",
        "test/mean/avg_score_pred_inactive/saliency",
        "test/mean/avg_score_pred_inactive/saliency-absolute",
        "test/mean/avg_score_pred_inactive/input_x_gradient",
    ]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_active" for i in range(20)]
    # args.track_metrics += ["test" + "/" + "smile" + str(i) + "/" + "avg_score_true_inactive" for i in range(20)]

    # attribution options
    args.attribution_kwargs = {
        "data_types": ["test"],
        "methods": [
            {"integrated_gradients": {
                "n_steps": 50,
                "postprocess": None
            }},
            {"saliency": {
                "postprocess": None,
                "abs": False,
            }},
            # {"saliency-absolute": {
            #     "postprocess": None,
            #     "abs": True,
            # }},
            {"input_x_gradient": {
                "postprocess": None
            }},
        ],
        "track_metrics": args.track_metrics,
        "label": "active_g10",
        "label_idx": 0,
        "references": Hergophores.ACTIVES_UNIQUE_,
        # "nr_samples": 100,
    }

    # trainer/logging args
    args.objective_name = "val/loss"
    args.minimize = True
    args.experiment_name = "herg_mlp_test2"
    args.checkpoint_objective = "val/loss"
    args.checkpoint_minimize = True
    args.tracking_uri = os.getenv("TRACKING_URI", default="http://localhost:5000")
    args.max_steps = 1000
    args.gradient_clip_val = 1.0
    args.stochastic_weight_avg = False
    args.seed = random.randint(0, 2 ** 32 - 1)
    args.patience = 10

    # data module args
    args.batch_size = 256
    # args.split_type = "random_kfold"
    # args.split_size = (5, 0, 1)
    args.split_type = "random"
    args.split_size = (0.6, 0.2, 0.2)
    args.split_seed = random.randint(0, 2 ** 32 - 1)
    args.standardize = False

    # args.use_labels = ["active_g10", "active_g20", "active_g40", "active_g60", "active_g80", "active_g100"]
    args.use_labels = ["active_g10"]

    args.featurizer_name = "combined"
    args.featurizer_kwargs = {
        "fold": 1024,
        "radius": 3,
        "return_count": True,
        "use_chirality": True,
        "use_features": True,
    }
    args.featurizer_n_jobs = 8

    # args.num_workers = multiprocessing.cpu_count()
    args.num_workers = 8
    args.cache_dir = "../../../" + "data/herg/"

    args.run_name = "mlp"

    # model args
    args.hidden_size = [32] * 5
    args.dropout = 0.1
    args.normalize_input = True
    args.batch_norm = True
    args.momentum = 0.1

    args.lr = 0.01
    args.optimizer = "adam"
    # args.optimizer = "adamw"
    # args.optimizer_params = {"weight_decay": 0.001}

    args.scheduler = "exponential_decay"
    args.scheduler_params = {"decay_step": 800, "decay_rate": 0.9}
    # args.scheduler = "linear_with_warmup"
    # args.scheduler_params = {"warmup_steps": 0.05}
    # args.scheduler = "none"

    args.log_sparsity = True
    # args.log_sparsity = "verbose"
    # args.log_parameters = False

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

    results = {}
    if "kfold" in args.split_type:
        kfold = Kfold(
            function=train_mlp_kfold,
            args=args,
            **vars(args)
        )
        results = kfold.train()

    else:
        results, *_ = train_mlp(args)

    print(results)
