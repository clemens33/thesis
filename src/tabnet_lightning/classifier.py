from argparse import ArgumentParser
from typing import Tuple, Optional, List, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection, Accuracy, AUROC

from tabnet import TabNet
from tabnet_lightning.utils import get_linear_schedule_with_warmup, get_exponential_decay_scheduler


class TabNetClassifier(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 decision_size: int,
                 num_classes: int,
                 nr_layers: int = 1,
                 nr_shared_layers: int = 1,
                 nr_steps: int = 1,
                 gamma: float = 1.0,
                 eps: float = 1e-5,
                 momentum: float = 0.01,
                 virtual_batch_size: int = 8,
                 #
                 lambda_sparse: float = 1e-4,
                 #
                 categorical_indices: Optional[List[int]] = None,
                 categorical_size: Optional[List[int]] = None,
                 embedding_dims: Optional[List[int]] = None,
                 #
                 lr: float = 1e-4,
                 optimizer: str = "adam",
                 optimizer_params: Optional[dict] = None,
                 scheduler: str = "none",
                 scheduler_params: Optional[dict] = None,
                 #
                 class_weights: Optional[List[float]] = None,
                 #
                 **kwargs
                 ):
        super(TabNetClassifier, self).__init__()

        self.lambda_sparse = lambda_sparse

        self.lr = lr

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

        self.categorical_indices = categorical_indices
        if self.categorical_indices is not None:
            self.embeddings = self._init_categorical_embeddings(categorical_size, embedding_dims)

        self.encoder = TabNet(input_size=input_size,
                              feature_size=feature_size,
                              decision_size=decision_size,
                              nr_layers=nr_layers,
                              nr_shared_layers=nr_shared_layers,
                              nr_steps=nr_steps,
                              gamma=gamma,
                              eps=eps,
                              momentum=momentum,
                              virtual_batch_size=virtual_batch_size,
                              **kwargs)

        self.classifier = nn.Linear(in_features=decision_size, out_features=num_classes)

        class_weights = torch.Tensor(class_weights) if class_weights is not None else None
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        metrics = MetricCollection([
            Accuracy(),
            AUROC(num_classes=num_classes, average="macro")  # TODO check -> leads to memory leak (atm fixed by calling reset in epoch end callbacks)
        ])

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.save_hyperparameters()

    def _init_categorical_embeddings(self, categorical_size: Optional[List[int]] = None,
                                     embedding_dims: Optional[List[int]] = None) -> nn.ModuleList:

        assert len(categorical_size) == len(
            embedding_dims), f"categorical_size length {len(categorical_size)} must be the same as embedding_dims length {len(embedding_dims)}"

        return nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=dim) for size, dim in zip(categorical_size, embedding_dims)
        ])

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.categorical_indices is not None:
            inputs = self._embeddings(inputs)

        decision, mask, entropy = self.encoder(inputs)

        logits = self.classifier(decision)

        return logits, mask, entropy

    def _embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        for idx, embedding in zip(self.categorical_indices, self.embeddings):
            input = inputs[..., idx].long()

            output = embedding(input)
            inputs[..., idx] = output[..., 0]

        return inputs

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, mask, entropy = self(inputs)
        preds = torch.softmax(logits, dim=-1)

        loss = self.loss_fn(logits, labels)
        loss = loss + entropy * self.lambda_sparse

        return loss, logits, preds, labels

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, preds, labels = self._step(*batch)

        self.log("train/loss", loss, prog_bar=True)

        output = self.train_metrics(preds, labels)
        self.log_dict(output)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, logits, preds, labels = self._step(*batch)

        self.log("val/loss", loss, prog_bar=True)

        output = self.val_metrics(preds, labels)
        self.log_dict(output, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_metrics.reset()

    def test_step(self, batch, batch_id):
        loss, logits, preds, labels = self._step(*batch)

        self.log("test/loss", loss)

        output = self.test_metrics(preds, labels)
        self.log_dict(output)

    def test_epoch_end(self, outputs: List[Any]):
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()

        if self.scheduler == "linear_with_warmup":
            if "warmup_steps" not in self.scheduler_params:
                raise KeyError(f"{self.scheduler_params} is missing warmup_steps - required for scheduler linear_with_warmup")

            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.scheduler_params["warmup_steps"], num_training_steps=self.max_steps
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        elif self.scheduler == "step_lr":
            if "decay_rate" not in self.scheduler_params or "decay_step" not in self.scheduler_params:
                raise KeyError(f"{self.scheduler_params} is missing decay_rate or decay_step - required for scheduler step_lr")

            scheduler = StepLR(optimizer, step_size=self.scheduler_params["decay_step"], gamma=self.scheduler_params["decay_rate"])
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        elif self.scheduler == "exponential_decay":
            if "decay_rate" not in self.scheduler_params or "decay_step" not in self.scheduler_params:
                raise KeyError(f"{self.scheduler_params} is missing decay_rate or decay_step - required for scheduler exponential_decay")

            scheduler = get_exponential_decay_scheduler(optimizer, decay_step=self.scheduler_params["decay_step"],
                                                        decay_rate=self.scheduler_params["decay_rate"])
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

    def _configure_optimizer(self) -> Optimizer:
        if self.optimizer == "adamw":
            if not "weight_decay" in self.optimizer_params:
                raise KeyError(f"{self.optimizer_params} is missing weight_decay - required for adamw")

            # remove decay from bias terms and batch normalization
            no_decay = ["bias", "bn.weight"]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.optimizer_params["weight_decay"],
                    "lr": self.lr,
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": self.lr,
                },
            ]

            optimizer = AdamW(optimizer_grouped_parameters)

            return optimizer
        elif self.optimizer == "adam":
            optimizer = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

            return optimizer
        else:
            raise ValueError(f"optimizer {self.optimizer} is not implemented")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.trainer.max_steps:
                self.max_steps = self.trainer.max_steps
            else:
                total_devices = self.trainer.num_gpus * self.trainer.num_nodes
                train_batches = len(self.train_dataloader()) // total_devices
                self.max_steps = (self.trainer.max_epochs * train_batches) // self.trainer.accumulate_grad_batches

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("TabNetClassifier")

        parser.add_argument("--input_size", type=int, help="input feature size/dimensions")
        parser.add_argument("--feature_size", type=int, help="feature/hidden size")
        parser.add_argument("--decision_size", type=int, help="decision/encoder size - output of tabnet before applying any specific task")
        parser.add_argument("--num_classes", type=int, help="output size/number of possible output classes")
        parser.add_argument("--nr_layers", type=int, default=1, help="number of independent layers per feature transformer")
        parser.add_argument("--nr_shared_layers", type=int, default=1, help="number of shared layers over all feature transformer")
        parser.add_argument("--nr_steps", type=int, default=1, help="number of tabnet steps")
        parser.add_argument("--gamma", type=float, default=1.0, help="gamma/relaxation parameter, larger values mean more relaxed "
                                                                     "behavior to reuse input features over subsequent steps")
        parser.add_argument("--eps", type=float, default=1e-5, help="for numerical stability calculating entropy used in regularization")
        parser.add_argument("--momentum", type=float, default=0.1, help="momentum for batch normalization")
        parser.add_argument("--virtual_batch_size", type=int, default=8, help="virtual batch size for ghost batch norm")

        return parent_parser
