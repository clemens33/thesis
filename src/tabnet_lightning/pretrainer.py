from typing import Tuple, Optional, List, Any, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection, Accuracy, AUROC

from tabnet import TabNet
from tabnet_lightning.metrics import Sparsity
from tabnet_lightning.utils import get_linear_schedule_with_warmup, get_exponential_decay_scheduler, StackedEmbedding, MultiEmbedding


class TabNetPretrainer(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 decision_size: int,
                 num_classes: int,
                 nr_layers: int = 1,
                 nr_shared_layers: int = 1,
                 nr_steps: int = 1,
                 gamma: float = 1.0,
                 #
                 momentum: float = 0.01,
                 virtual_batch_size: int = 8,
                 normalize_input: bool = True,
                 #
                 lambda_sparse: float = 1e-4,
                 eps: float = 1e-5,
                 #
                 categorical_indices: Optional[List[int]] = None,
                 categorical_size: Optional[List[int]] = None,
                 embedding_dims: Optional[Union[List[int], int]] = None,
                 #
                 lr: float = 1e-4,
                 optimizer: str = "adam",
                 optimizer_params: Optional[dict] = None,
                 scheduler: str = "none",
                 scheduler_params: Optional[dict] = None,
                 #
                 class_weights: Optional[List[float]] = None,
                 #
                 log_sparsity: str = None,
                 #
                 **kwargs
                 ):
        """
        tabnet lightning classifier init function

        Args:
            input_size:  input feature size/dimensions
            feature_size: feature/hidden size
            decision_size: decision/encoder size - output of tabnet before applying any specific task
            num_classes: output size/number of possible output classes
            nr_layers:  number of independent layers per feature transformer
            nr_shared_layers: number of shared layers over all feature transformer
            nr_steps: number of tabnet steps
            gamma: gamma/relaxation parameter, larger values mean more relaxed, behavior to reuse input features over subsequent steps
            momentum: momentum for batch normalization
            virtual_batch_size: virtual batch size for ghost batch norm - if set to -1 no batch norm is applied in intermediate layers
            normalize_input: if batch norm is applied on input features
            lambda_sparse: sparsity regularization
            eps: for numerical stability calculating entropy used in regularization
            categorical_indices: which indices in the input features applies to categorical variables
            categorical_size: each categorical variables size
            embedding_dims: embedding dimension for each categorical variable
            lr: learning rate
            optimizer: optimizer name
            optimizer_params: optimizer params
            scheduler: scheduler name
            scheduler_params: scheduler params
            class_weights: optional class weights
            **kwargs: optional tabnet parameters
        """
        super(TabNetPretrainer, self).__init__()

        self.num_classes = num_classes
        self.lambda_sparse = lambda_sparse

        self.lr = lr

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

        self.categorical_indices = categorical_indices
        if self.categorical_indices is not None:
            if isinstance(embedding_dims, int) and embedding_dims == 1:
                # much faster version - only supports embedding_dim of 1 atm
                self.embeddings = StackedEmbedding(embedding_indices=categorical_indices, num_embeddings=categorical_size,
                                                   embedding_dim=embedding_dims)
            else:
                self.embeddings = MultiEmbedding(embedding_indices=categorical_indices, num_embeddings=categorical_size,
                                                 embedding_dims=embedding_dims)
        else:
            self.embeddings = nn.Identity()

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
                              normalize_input=normalize_input,
                              **kwargs)

        self.decoder =


        if num_classes == 2:
            self.classifier = nn.Linear(in_features=decision_size, out_features=1)
        else:
            self.classifier = nn.Linear(in_features=decision_size, out_features=num_classes)

        class_weights = torch.Tensor(class_weights) if class_weights is not None else None

        if num_classes == 2:
            pos_weight = class_weights[1] / class_weights.sum() if class_weights is not None else None
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        metrics = MetricCollection([
            Accuracy(),
            AUROC() if num_classes == 2 else AUROC(num_classes=num_classes),
            # TODO check -> leads to memory leak (atm fixed by calling reset in epoch end callbacks)
        ])

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.log_sparsity = str(log_sparsity)
        self._init_sparsity_metrics(self.log_sparsity, nr_steps)

        self.save_hyperparameters()

    def _init_sparsity_metrics(self, log_sparsity: str, nr_steps: int):

        if log_sparsity in ["verbose", "True"]:
            metrics = []
            # all (inputs + mask + all masks)
            metrics = [Sparsity() for _ in range(nr_steps + 3)]

            self.sparsity_metrics = nn.ModuleDict({
                "train_sparsity_metrics": nn.ModuleList([m.clone() for m in metrics]) if metrics else None,
                "val_sparsity_metrics": nn.ModuleList([m.clone() for m in metrics]) if metrics else None,
                "test_sparsity_metrics": nn.ModuleList([m.clone() for m in metrics]) if metrics else None,
            })

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        inputs = self.embeddings(inputs)

        decision, mask, entropy, decisions, masks = self.encoder(inputs)

        logits = self.classifier(decision)

        return logits, mask, entropy, decisions, masks

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        logits, mask, entropy, _, masks = self(inputs)

        if self.num_classes == 2:
            logits = logits.squeeze()
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        loss = self.loss_fn(logits, labels.float() if self.num_classes == 2 else labels)
        loss = loss + entropy * self.lambda_sparse

        return loss, logits, probs, labels, mask, masks

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, probs, labels, mask, masks = self._step(*batch)

        self.log("train/loss", loss, prog_bar=True)

        output = self.train_metrics(probs, labels)
        self.log_dict(output)

        self._log_sparsity(inputs=batch[0], mask=mask, masks=masks, prefix="train")

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        self.train_metrics.reset()

        self._log_sparsity(prefix="train", compute=True)

    def validation_step(self, batch, batch_idx):
        loss, logits, probs, labels, mask, masks = self._step(*batch)

        self.log("val/loss", loss, prog_bar=True)

        output = self.val_metrics(probs, labels)
        self.log_dict(output, prog_bar=True)

        self._log_sparsity(inputs=batch[0], mask=mask, masks=masks, prefix="val")

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_metrics.reset()

        self._log_sparsity(prefix="val", compute=True)

    def test_step(self, batch, batch_id):
        loss, logits, probs, labels, mask, masks = self._step(*batch)

        self.log("test/loss", loss)

        output = self.test_metrics(probs, labels)
        self.log_dict(output)

        self._log_sparsity(inputs=batch[0], mask=mask, masks=masks, prefix="test")

    def test_epoch_end(self, outputs: List[Any]):
        self.test_metrics.reset()

        self._log_sparsity(prefix="test", compute=True)

    def _log_sparsity(self, prefix: str, compute: bool = False, inputs: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                      masks: Optional[List[torch.Tensor]] = None):

        if self.log_sparsity in ["verbose", "True"]:
            metrics = self.sparsity_metrics[prefix + "_sparsity_metrics"]

            if compute:
                self.log(prefix + "/sparsity_inputs", metrics[0].compute(), on_step=False, on_epoch=True)
                self.log(prefix + "/sparsity_mask", metrics[1].compute(), on_step=False, on_epoch=True)
                self.log(prefix + "/sparsity_masks_sum", metrics[2].compute(), on_step=False, on_epoch=True)
            else:
                masks_sum = torch.stack(masks, dim=0).sum(dim=0)

                metrics[0].update(inputs)
                metrics[1].update(mask)
                metrics[2].update(masks_sum)

            if self.log_sparsity == "verbose":
                for i, metric in enumerate(metrics[3:]):
                    if compute:
                        self.log(prefix + "/sparsity_mask_step" + str(i), metric.compute(), on_step=False, on_epoch=True)
                    else:
                        metric.update(masks[i])

            if compute:
                for metric in metrics:
                    metric.reset()

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
