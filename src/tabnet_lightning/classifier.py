from argparse import ArgumentParser
from typing import Tuple, Union, Optional, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW

from tabnet import TabNet
from tabnet_lightning.utils import get_linear_schedule_with_warmup

from torchmetrics import MetricCollection, Accuracy, Precision, Recall


class TabNetClassifier(pl.LightningModule):
    def __init__(self,
                 #
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
                 lr: float = 1e-4,
                 weight_decay: float = 1e-3,
                 scheduler: str = "none",
                 num_warmup_steps: Union[int, float] = 0.1,
                 #
                 class_weights: Optional[List[float]] = None,
                 #
                 **kwargs
                 ):
        super(TabNetClassifier, self).__init__()

        self.lambda_sparse = lambda_sparse

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.scheduler = scheduler

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
            # Precision(num_classes=num_classes, average="macro"),
            # Recall(num_classes=num_classes, average="macro")
        ])

        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decision, mask, entropy = self.encoder(inputs)

        logits = self.classifier(decision)

        return logits, mask, entropy

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _, entropy = self(inputs)
        preds = torch.softmax(logits, dim=-1)

        loss = self.loss_fn(logits, labels)
        loss = loss + entropy * self.lambda_sparse

        return loss, logits, preds, labels

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, preds, labels = self._step(*batch)

        output = self.train_metrics(preds, labels)
        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, preds, labels = self._step(*batch)

        self.log("val/loss", loss, prog_bar=True)

        output = self.val_metrics(preds, labels)
        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.lr,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        if self.scheduler == "linear_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

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
