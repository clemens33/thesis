import random
import sys

import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import MLFlowLogger

from sklearn.metrics import roc_auc_score
from torchmetrics.functional import auroc

from datasets import MolNetClassifierDataModule
from tabnet_lightning import TabNetClassifier, TabNetTrainer
from baseline import MLPClassifier

PATH = "./26/c6fb941dd2594847b2908b0dcaaa54aa/checkpoints/epoch=5-step=41.ckpt"
INIT_SEED = 1
TRIALS = 50


def main():
    seed = INIT_SEED

    mlf_logger = MLFlowLogger(
        experiment_name="bbbp_bl_test1",
        tracking_uri="https://mlflow.kriechbaumer.at"
    )

    classifier = MLPClassifier.load_from_checkpoint(PATH)
    classifier.eval()

    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="trainable_parameters",
                                    value=sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    exp = mlf_logger.experiment
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="init_seed", value=INIT_SEED)
    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="path", value=PATH)

    seed_everything(seed)
    path = "../../../"

    dm = MolNetClassifierDataModule(
        name="bbbp",
        batch_size=512,
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

        logger=mlf_logger
    )
    trainer.log_hyperparameters(mlf_logger)

    aurocs = []
    for i in range(TRIALS):
        mlf_logger.experiment.log_metric(run_id=mlf_logger.run_id, key="seed", value=seed, step=i)

        # # we only have one batch - no iteration needed
        # with torch.no_grad():
        #     inputs, y_true = next(iter(dm.test_dataloader()))
        #     logits, mask, entropy = classifier(inputs)
        #
        #     logits = logits.squeeze()
        #     y_scores = torch.sigmoid(logits)
        #
        #     auroc_value = roc_auc_score(y_true, y_scores)

        mlf_logger.experiment.log_metric(run_id=mlf_logger.run_id, key="seed", value=seed, step=i)

        results = trainer.test(model=classifier, datamodule=dm)[0]
        aurocs.append(results["test/AUROC"])
        # aurocs.append(auroc_value)

        seed = random.randint(0, 2 ** 32 - 1)

        dm = MolNetClassifierDataModule(
            name="bbbp",
            batch_size=512,
            seed=seed,
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

        mlf_logger.experiment.log_metric(run_id=mlf_logger.run_id, key="auroc_mean", value=np.array(aurocs).mean(), step=i)
        mlf_logger.experiment.log_metric(run_id=mlf_logger.run_id, key="auroc_std", value=np.array(aurocs).std(), step=i)

    mlf_logger.experiment.end_run()


if __name__ == "__main__":
    main()
