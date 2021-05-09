from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.cli import LightningCLI

from datasets import MolNetClassifierDataModule
from tabnet_lightning import TabNetClassifier


def main():
    mlf_logger = MLFlowLogger(
        experiment_name="bbbp5",
        tracking_uri="https://mlflow.kriechbaumer.at"
    )

    exp = mlf_logger.experiment
    #mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key="seed", value=seed)

    cli = LightningCLI(TabNetClassifier, MolNetClassifierDataModule)

    print("done")


if __name__ == '__main__':
    main()
