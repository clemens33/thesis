from typing import List, Union, Optional, Tuple, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection, Accuracy, AUROC

from tabnet_lightning.utils import get_linear_schedule_with_warmup, get_exponential_decay_scheduler


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: Union[List[int], int],
                 output_size: int,
                 activation: nn.Module = nn.ReLU(),
                 normalize_input: bool = False,
                 batch_norm: bool = False
                 ):
        super(MLP, self).__init__()

        self.input_norm = nn.BatchNorm1d(input_size) if normalize_input else nn.Identity()

        hidden_sizes = [hidden_size] if isinstance(hidden_size, int) else hidden_size

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hidden_size),
                activation,
                nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity()
            ))

            in_features = hidden_size

        self.layers = nn.ModuleList(layers)

        self.out = nn.Linear(in_features=hidden_sizes[-1], out_features=output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.input_norm(inputs)

        for layer in self.layers:
            inputs = layer(inputs)

        output = self.out(inputs)

        return output


class MLPClassifier(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: Union[List[int], int],
                 num_classes: int,
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

        super(MLPClassifier, self).__init__()

        self.num_classes = num_classes

        output_size = 1 if num_classes == 2 else num_classes
        self.classifier = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

        self.lr = lr

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

        self.categorical_indices = categorical_indices
        if self.categorical_indices is not None:
            self.embeddings = self._init_categorical_embeddings(categorical_size, embedding_dims)

        class_weights = torch.Tensor(class_weights) if class_weights is not None else None

        if num_classes == 2:
            pos_weight = class_weights[1] / class_weights.sum() if class_weights is not None else None
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        metrics = MetricCollection([
            # Accuracy(num_classes=num_classes),
            # AUROC(num_classes=num_classes, average="macro"),
            Accuracy(),
            AUROC(average="macro"),
            # TODO check -> leads to memory leak (atm fixed by calling reset in epoch end callbacks)
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.categorical_indices is not None:
            inputs = self._embeddings(inputs)

        logits = self.classifier(inputs)

        return logits

    def _embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        for idx, embedding in zip(self.categorical_indices, self.embeddings):
            input = inputs[..., idx].long()

            output = embedding(input)
            inputs[..., idx] = output[..., 0]

        return inputs

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(inputs)

        if self.num_classes == 2:
            logits = logits.squeeze()
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        loss = self.loss_fn(logits, labels.float() if self.num_classes == 2 else labels)

        return loss, logits, probs, labels

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, probs, labels = self._step(*batch)

        self.log("train/loss", loss, prog_bar=True)

        output = self.train_metrics(probs, labels)
        self.log_dict(output)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, logits, probs, labels = self._step(*batch)

        self.log("val/loss", loss, prog_bar=True)

        output = self.val_metrics(probs, labels)
        self.log_dict(output, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_metrics.reset()

    def test_step(self, batch, batch_id):
        loss, logits, probs, labels = self._step(*batch)

        self.log("test/loss", loss)

        output = self.test_metrics(probs, labels)
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
