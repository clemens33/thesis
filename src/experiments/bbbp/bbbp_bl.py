import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from datasets import MolNetClassifierDataModule
from tabnet_lightning import TabNetTrainer
from baseline import MLPClassifier


def main():
    seed = 1234

    mlf_logger = MLFlowLogger(
        experiment_name="bbbp_bl_mlp",
        tracking_uri="https://mlflow.kriechbaumer.at"
    )

    exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=seed)

    seed_everything(seed)
    path = "../../../"

    dm = MolNetClassifierDataModule(
        name="bbbp",
        batch_size=256,
        seed=seed,
        split="random",
        split_size=(0.8, 0.1, 0.1),
        radius=6,
        n_bits=4096,
        chirality=True,
        features=True,
        num_workers=0,  # 0 for debugging
        cache_dir=path + "data/molnet/bbbp/",
        use_cache=True
    )

    dm.prepare_data()
    dm.setup()

    dm.log_hyperparameters(mlf_logger)

    classifier = MLPClassifier(
        input_size=dm.input_size,
        hidden_size=[512, 512, 512],
        num_classes=2,
        lr=0.001,
        optimizer="adam",
        # scheduler="exponential_decay",
        # scheduler_params={"decay_step": 100, "decay_rate": 0.95},

        # optimizer="adamw",
        # optimizer_params={"weight_decay": 0.0001},
        scheduler="linear_with_warmup",
        # scheduler_params={"warmup_steps": 0.1},
        scheduler_params={"warmup_steps": 10},

        class_weights=dm.class_weights,
    )
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
                                    value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    checkpoint_callback = ModelCheckpoint(
        monitor="val/AUROC",
        # dirpath='my/path/',
        # filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
        # save_top_k=3,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = TabNetTrainer(
        # default_root_dir=path + "logs/molnet/bbbp2/",

        gpus=1,
        # checkpoint_callback=False,
        # accelerator="ddp",

        max_steps=1000,
        # max_epochs=300,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=-1,

        fast_dev_run=False,
        deterministic=True,
        # precision=16,

        # gradient_clip_algorithm="value",
        # gradient_clip_val=2,

        callbacks=[lr_monitor, checkpoint_callback],
        logger=mlf_logger
    )
    trainer.log_hyperparameters(mlf_logger)

    trainer.fit(classifier, dm)

    trainer.test(model=classifier, datamodule=dm)


if __name__ == "__main__":
    main()
