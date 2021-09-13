from typing import Tuple, Optional, List, Any, Union, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection, Accuracy, AUROC

from tabnet import TabNet
from tabnet_lightning.metrics import Sparsity, CustomAccuracy, CustomAUROC
from tabnet_lightning.utils import get_linear_schedule_with_warmup, get_exponential_decay_scheduler, StackedEmbedding, MultiEmbedding, \
    plot_masks, plot_rankings, replace_key_name


class ClassificationHead(nn.Module):
    def __init__(self, input_size: int, num_classes: Union[int, List[int]], class_weights: Optional[List[float]] = None,
                 ignore_index: int = -100):
        super().__init__()

        self.input_size = input_size
        self.ignore_index = ignore_index

        if isinstance(num_classes, int):
            if num_classes == 2:
                self.objective = "binary"
                self.num_classes = 2
                self.num_targets = 1
            elif num_classes > 2:
                self.objective = "multi-class"
                self.num_classes = num_classes
                self.num_targets = 1
            else:
                ValueError(f"num_classes {num_classes} not supported")
        elif isinstance(num_classes, list):
            if all(c == 2 for c in num_classes):
                self.objective = "binary-multi-target"
                self.num_classes = 2
                self.num_targets = len(num_classes)
            else:
                raise AttributeError("multi class (non binary), multi target objective not supported yet")
        else:
            raise AttributeError(f"provided num classes type {type(num_classes)} not supported")

        class_weights = torch.Tensor(class_weights) if class_weights is not None else None

        if self.objective == "binary":
            self.classifier = nn.Linear(in_features=input_size, out_features=1)

            pos_weight = None
            if class_weights:
                if len(class_weights) == 2:
                    pos_weight = class_weights[1] / class_weights.sum()
                elif len(class_weights) == 1:
                    pos_weight = class_weights[0]
                else:
                    raise AttributeError(f"provided class weights {len(class_weights)} do not match binary classification objective")

            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        elif self.objective == "multi-class":
            self.classifier = nn.Linear(in_features=input_size, out_features=num_classes)

            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

        elif self.objective == "binary-multi-target":
            if class_weights:
                if len(class_weights) != len(num_classes):
                    raise AttributeError(
                        f"length of provided class weights {len(class_weights)} does not match the provided number of classes {num_classes}")

            self.classifier = nn.Linear(in_features=input_size, out_features=len(num_classes))

            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=class_weights)

    def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, Union[None, torch.Tensor]]:

        logits, probs, loss = None, None, None
        if self.objective == "binary":
            logits = self.classifier(inputs)

            logits = logits.squeeze()
            probs = torch.sigmoid(logits)

            loss = self.loss_fn(logits, labels.float()) if labels is not None else None
        elif self.objective == "multi-class":
            logits = self.classifier(inputs)

            probs = torch.softmax(logits, dim=-1)

            loss = self.loss_fn(logits, labels) if labels is not None else None
        elif self.objective == "binary-multi-target":
            logits = self.classifier(inputs)
            probs = torch.sigmoid(logits)

            if labels is not None:
                mask = torch.where(labels == self.ignore_index, 0, 1)
                labels = labels * mask

                loss = self.loss_fn(logits, labels.float())

                loss = loss * mask
                loss = loss.mean()

        return logits, probs, loss


class TabNetClassifier(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 decision_size: int,
                 num_classes: Union[int, List[int]],
                 nr_layers: int = 1,
                 nr_shared_layers: int = 1,
                 nr_steps: int = 1,
                 momentum: float = 0.1,
                 virtual_batch_size: int = 8,
                 normalize_input: bool = True,
                 eps: float = 1e-5,
                 gamma: float = 1.0,
                 relaxation_type: str = "gamma_fixed",
                 alpha: float = 2.0,
                 attentive_type: str = "sparsemax",
                 #
                 lambda_sparse: float = 1e-4,
                 #
                 ignore_index: int = -100,
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
                 log_parameters: bool = True,
                 log_metrics: bool = True,
                 #
                 plot_masks: Optional[dict] = None,
                 plot_rankings: Optional[dict] = None,
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
            log_sparsity: optional str - if "True" logs the sparsity of the aggregated mask - if set to "verbose" logs all masks for each step
            log_parameters: optional - if True logs the parameter alpha and gamma
            **kwargs: optional tabnet parameters
        """
        super(TabNetClassifier, self).__init__()

        if plot_rankings is None:
            plot_rankings = {}
        if plot_masks is None:
            plot_masks = {}

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

                              relaxation_type=relaxation_type,
                              alpha=alpha,
                              attentive_type=attentive_type,
                              **kwargs)

        self.classifier = ClassificationHead(input_size=decision_size, num_classes=num_classes, class_weights=class_weights,
                                             ignore_index=ignore_index)

        metrics = MetricCollection(
            [
                CustomAccuracy(num_targets=self.classifier.num_targets, ignore_index=ignore_index),
                CustomAUROC(num_targets=self.classifier.num_targets, ignore_index=ignore_index, return_verbose=True),
            ]
            if self.classifier.objective == "binary-multi-target" else [
                Accuracy(),
                AUROC(num_classes=self.classifier.num_classes)
            ]
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.log_sparsity = str(log_sparsity)
        self._init_sparsity_metrics(self.log_sparsity, nr_steps)

        self.log_parameters = log_parameters
        self.log_metrics = log_metrics

        self.plot_masks = plot_masks
        self.plot_rankings = plot_rankings

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

    def _postprocess_metric_output(self, output: Dict) -> Dict:
        """helper function to process tuple metric output - atm for CustomAUROC output"""

        output = replace_key_name(output, "Custom", "")

        _output = {}
        for k, v in output.items():
            if isinstance(v, tuple) and "AUROC" in k:
                auroc, aurocs, thresholds = v
                _output[k] = auroc

                for i in range(len(aurocs)):
                    _output[k + "-t" + str(i)] = aurocs[i]

                for i in range(len(thresholds)):
                    _output[k.replace("AUROC", "threshold") + "-t" + str(i)] = thresholds[i]
            else:
                _output[k] = v

        return _output

    def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, Union[None, torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        inputs = self.embeddings(inputs)

        decision, mask, entropy, decisions, masks = self.encoder(inputs)

        logits, probs, loss = self.classifier(decision, labels)

        return logits, probs, loss, mask, entropy, decisions, masks

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:

        logits, probs, loss, mask, entropy, _, masks = self(inputs, labels)

        loss = loss + entropy * self.lambda_sparse

        return loss, logits, probs, labels, mask, masks

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, logits, probs, labels, mask, masks = self._step(*batch)

        self.log("train/loss", loss, prog_bar=True)

        output = self.train_metrics(probs, labels)
        self.log_dict(self._postprocess_metric_output(output))

        self._log_sparsity(inputs=batch[0], mask=mask, masks=masks, prefix="train")
        self._log_parameters()

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        self.train_metrics.reset()

        self._log_sparsity(prefix="train", compute=True)

    def validation_step(self, batch, batch_idx):
        loss, logits, probs, labels, mask, masks = self._step(*batch)

        self.log("val/loss", loss, prog_bar=True)

        output = self.val_metrics(probs, labels)

        if self.log_metrics:
            self.log_dict(self._postprocess_metric_output(output), prog_bar=False)

        self._log_sparsity(inputs=batch[0], mask=mask, masks=masks, prefix="val")

        if self.plot_masks is not None:
            return {
                "inputs": batch[0],
                "labels": batch[1],
                "mask": mask,
                "masks": masks,
            }

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_metrics.reset()

        self._log_sparsity(prefix="val", compute=True)

    def test_step(self, batch, batch_id):
        loss, logits, probs, labels, mask, masks = self._step(*batch)

        self.log("test/loss", loss)

        output = self.test_metrics(probs, labels)

        if self.log_metrics:
            self.log_dict(self._postprocess_metric_output(output))

        self._log_sparsity(inputs=batch[0], mask=mask, masks=masks, prefix="test")

        if self.plot_masks is not None:
            return {
                "inputs": batch[0],
                "labels": batch[1],
                "mask": mask,
                "masks": masks,
            }

    def test_epoch_end(self, outputs: List[Any]):
        self.test_metrics.reset()

        self._log_sparsity(prefix="test", compute=True)

        if self.plot_masks.get("on_test_epoch_end", False):
            self._plot_masks(outputs)

        if self.plot_rankings.get("on_test_epoch_end", False):
            self._plot_rankings(outputs)

    def _plot_rankings(self, outputs: List[Any]):
        if self.plot_rankings:
            top_k = self.plot_rankings.get("top_k", None)
            feature_names = self.plot_rankings.get("feature_names", None)
            out_fname = self.plot_rankings.get("out_fname", None)

            mask = torch.cat([o["mask"] for o in outputs], dim=0)

            kwargs = {"out_fname": out_fname} if out_fname else {}

            path = plot_rankings(mask, feature_names=feature_names, top_k=top_k, **kwargs)
            run_id = self.logger.run_id
            self.logger.experiment.log_artifact(run_id=run_id, local_path=path)

    def _plot_masks(self, outputs: List[Any]):
        if self.plot_masks is not None:

            nr_samples = self.plot_masks.get("nr_samples", 20)
            normalize_inputs = self.plot_masks.get("normalize_inputs", False)
            verbose = self.plot_masks.get("verbose", False)
            feature_names = self.plot_masks.get("feature_names", None)
            out_fname = self.plot_masks.get("out_fname", None)

            inputs = torch.cat([o["inputs"] for o in outputs], dim=0)
            labels = torch.cat([o["labels"] for o in outputs], dim=0)
            mask = torch.cat([o["mask"] for o in outputs], dim=0)

            masks = []
            for i in range(len(outputs[0]["masks"])):
                masks.append(torch.cat([o["masks"][i] for o in outputs], dim=0))

            kwargs = {"out_fname": out_fname} if out_fname else {}
            path = plot_masks(inputs=inputs,
                              aggregated_mask=mask,
                              labels=labels,
                              masks=masks if verbose else None,
                              nr_samples=nr_samples,
                              feature_names=feature_names,
                              normalize_inputs=normalize_inputs,
                              **kwargs)

            run_id = self.logger.run_id
            self.logger.experiment.log_artifact(run_id=run_id, local_path=path)

    def _log_parameters(self):
        """logs the gamma, alphas and other parameters if available"""
        if self.log_parameters:

            gammas, alphas, slopes = [], [], []
            for i, step in enumerate(self.encoder.steps):
                gammas.append(step.attentive_transformer.gamma)
                alphas.append(step.attentive_transformer.attentive.alpha)

                if step.attentive_transformer.attentive.attentive_type == "binary_mask":
                    slopes.append(step.attentive_transformer.attentive.attentive.activation.slope)
                else:
                    slopes.append(0.0)

            self.log("gamma", torch.mean(torch.Tensor(gammas)), on_step=True, on_epoch=False)
            self.log("alpha", torch.mean(torch.Tensor(alphas)), on_step=True, on_epoch=False)
            self.log("slope", torch.mean(torch.Tensor(slopes)), on_step=True, on_epoch=False)

            # only log individual gammas and alphas if they are not the same
            if gammas[0] != gammas[-1]:
                for i, gamma in enumerate(gammas):
                    self.log("gamma_step" + str(i), gamma, on_step=True, on_epoch=False)

            if alphas[0] != alphas[-1]:
                for i, alpha in enumerate(alphas):
                    self.log("alpha_step" + str(i), alpha, on_step=True, on_epoch=False)

            if slopes[0] != slopes[-1]:
                for i, slope in enumerate(slopes):
                    self.log("slope_step" + str(i), slope, on_step=True, on_epoch=False)

    def _log_sparsity(self, prefix: str, compute: bool = False, inputs: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                      masks: Optional[List[torch.Tensor]] = None):

        if self.log_sparsity in ["verbose", "True"]:
            metrics = self.sparsity_metrics[prefix + "_sparsity_metrics"]

            if compute:
                self.log(prefix + "/sparsity_inputs", metrics[0].compute(), on_step=False, on_epoch=True)
                self.log(prefix + "/sparsity_mask", metrics[1].compute(), on_step=False, on_epoch=True)
                # self.log(prefix + "/sparsity_masks_sum", metrics[2].compute(), on_step=False, on_epoch=True)
            else:
                # masks_sum = torch.stack(masks, dim=0).sum(dim=0)

                metrics[0].update(inputs)
                metrics[1].update(mask)
                # metrics[2].update(masks_sum)

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
