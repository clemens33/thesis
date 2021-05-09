import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from datasets import MolNetClassifierDataModule
from tabnet_lightning import TabNetClassifier, TabNetTrainer


def main():
    seed = 1

    mlf_logger = MLFlowLogger(
        experiment_name="bbbp7",
        tracking_uri="https://mlflow.kriechbaumer.at"
    )

    exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=seed)

    seed_everything(seed)
    path = "../../../"

    dm = MolNetClassifierDataModule(
        name="bbbp",
        batch_size=128,
        seed=seed,
        split="random",
        split_size=(0.8, 0.1, 0.1),
        radius=6,
        n_bits=4096,
        chirality=True,
        features=True,
        num_workers=8, # 0 for debugging
        cache_dir=path + "data/molnet/bbbp/",
        use_cache=True
    )

    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    classifier = TabNetClassifier(
        input_size=dm.input_size,
        feature_size=64,
        decision_size=32,
        num_classes=len(dm.classes),
        nr_layers=2,
        nr_shared_layers=2,
        nr_steps=14,
        gamma=3.0,

        # pytorch batch norm uses 1 - momentum (e.g. 0.3 pytorch is the same as tf 0.7)
        # momentum=0.01,
        virtual_batch_size=-1,  # -1 do not use any batch normalization
        # virtual_batch_size=32,
        normalize_input=False,

        # decision_activation=torch.tanh,

        lambda_sparse=0.00,

        # define embeddings for categorical variables - otherwise raw value is used
        # categorical_indices=list(range(dm.input_size)),
        # categorical_size=[2] * dm.input_size,
        # embedding_dims=[1] * dm.input_size,

        lr=0.01,
        # optimizer="adam",
        # scheduler="exponential_decay",
        # scheduler_params={"decay_step": 100, "decay_rate": 0.95},

        optimizer="adamw",
        optimizer_params={"weight_decay": 0.0001},
        scheduler="linear_with_warmup",
        scheduler_params={"warmup_steps": 0.1},

        class_weights=dm.class_weights,
    )
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters", value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))


    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = TabNetTrainer(
        # default_root_dir=path + "logs/molnet/bbbp2/",

        gpus=1,
        checkpoint_callback=False,
        # accelerator="ddp",

        max_steps=500,
        # max_epochs=300,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=-1,

        fast_dev_run=False,
        deterministic=True,
        # precision=16,

        # gradient_clip_algorithm="value",
        # gradient_clip_val=2,

        callbacks=[lr_monitor],
        logger=mlf_logger
    )
    trainer.log_hyperparameters(mlf_logger)

    trainer.fit(classifier, dm)

    trainer.test(model=classifier, datamodule=dm)


if __name__ == "__main__":
    main()
