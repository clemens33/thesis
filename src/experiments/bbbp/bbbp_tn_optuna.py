import optuna

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from datasets import MolNetClassifierDataModule
from tabnet_lightning import TabNetTrainer, TabNetClassifier

SEED = 987654321
#BATCH_SIZE = 256
MAX_STEPS = 2000


def objective(trial: optuna.trial.Trial) -> float:
    mlf_logger = MLFlowLogger(
        experiment_name="bbbp_optuna7",
        tracking_uri="https://mlflow.kriechbaumer.at"
    )

    exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=SEED)

    seed_everything(SEED)
    path = "../../../"

    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])

    dm = MolNetClassifierDataModule(
        name="bbbp",
        batch_size=batch_size,
        seed=SEED,
        split="random",
        split_size=(0.8, 0.1, 0.1),
        radius=6,
        n_bits=4096,
        chirality=True,
        features=True,
        num_workers=8,  # 0 for debugging
        cache_dir=path + "data/molnet/bbbp/",
        use_cache=True
    )

    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    # parameter search space
    # decision_size = trial.suggest_int("decision_size", 8, 64)
    # nr_layers = trial.suggest_int("nr_layers", 0, 4)
    # nr_shared_layers = trial.suggest_int("nr_shared_layers", 1, 2)
    # nr_steps = trial.suggest_int("nr_steps", 1, 100)
    # gamma = trial.suggest_float("gamma", 1.0, 5.0, step=0.2)
    # lambda_sparse = trial.suggest_categorical("lambda_sparse", [0, 1e-6, 1e-4, 1e-3, 0.01, 0.1])
    # lr = trial.suggest_float("lr", 1e-6, 0.01, log=True)

    decision_size = trial.suggest_categorical("decision_size", [8, 16, 24, 32, 64, 128])
    nr_steps = trial.suggest_categorical("nr_steps", [3, 4, 5, 6, 7, 8, 9, 10])
    gamma = trial.suggest_categorical("gamma", [1.0, 1.2, 1.5, 2.0])
    lambda_sparse = trial.suggest_categorical("lambda_sparse", [0, 1e-6, 1e-4, 1e-3, 0.01, 0.1])

    lr = trial.suggest_categorical("lr", [0.005, 0.01, 0.02, 0.025])
    decay_step = trial.suggest_categorical("decay_step", [5, 20, 80, 100])
    decay_rate = trial.suggest_categorical("decay_rate", [0.4, 0.8, 0.9, 0.95])

    classifier = TabNetClassifier(
        input_size=dm.input_size,
        feature_size=decision_size * 2,
        decision_size=decision_size,
        num_classes=len(dm.classes),
        nr_layers=2,
        nr_shared_layers=2,
        nr_steps=nr_steps,
        gamma=gamma,

        # pytorch batch norm uses 1 - momentum (e.g. 0.3 pytorch is the same as tf 0.7)
        # momentum=0.01,
        virtual_batch_size=-1,  # -1 do not use any batch normalization
        # virtual_batch_size=32,
        normalize_input=False,

        # decision_activation=torch.tanh,

        lambda_sparse=lambda_sparse,

        # define embeddings for categorical variables - otherwise raw value is used
        # categorical_indices=list(range(dm.input_size)),
        # categorical_size=[2] * dm.input_size,
        # embedding_dims=[1] * dm.input_size,

        lr=lr,
        optimizer="adam",
        scheduler="exponential_decay",
        scheduler_params={"decay_step": decay_step, "decay_rate": decay_rate},

        #optimizer="adamw",
        #optimizer_params={"weight_decay": 0.0001},
        #scheduler="linear_with_warmup",
        #scheduler_params={"warmup_steps": 0.1},

        class_weights=dm.class_weights,
    )
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
                                    value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    early_stopping = EarlyStopping(monitor="val/AUROC", patience=20, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    optuna_pruner = PyTorchLightningPruningCallback(trial, monitor="val/AUROC")

    trainer = TabNetTrainer(
        # default_root_dir=path + "logs/molnet/bbbp2/",

        gpus=1,
        checkpoint_callback=False,
        # accelerator="ddp",

        max_steps=MAX_STEPS,
        # max_epochs=MAX_STEPS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=-1,

        fast_dev_run=False,
        deterministic=True,
        # precision=16,

        # gradient_clip_algorithm="value",
        # gradient_clip_val=2,

        callbacks=[lr_monitor, early_stopping, optuna_pruner],
        #callbacks=[lr_monitor, optuna_pruner],
        logger=mlf_logger
    )
    trainer.log_hyperparameters(mlf_logger)

    trainer.fit(classifier, dm)

    trainer.test(model=classifier, datamodule=dm)

    return trainer.callback_metrics["val/AUROC"].item()


if __name__ == '__main__':
    #pruner = optuna.pruners.SuccessiveHalvingPruner()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10, interval_steps=1)

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=3600 * 2, catch=(ValueError, ))
