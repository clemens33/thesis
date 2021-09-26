from functools import wraps
from timeit import default_timer as timer
from typing import Union, List, Optional, Tuple

import torch
from torch import nn as nn
from torch.optim.lr_scheduler import LambdaLR


class ClassificationHead(nn.Module):
    def __init__(self, input_size: int, num_classes: Union[int, List[int]], class_weights: Optional[List[float]] = None,
                 ignore_index: int = -100):
        super().__init__()

        self.input_size = input_size
        self.ignore_index = ignore_index

        self.objective, self.num_classes, self.num_targets = determine_objective(num_classes)

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
                labels = labels.unsqueeze(dim=1) if labels.ndim == 1 else labels
                mask = torch.where(labels == self.ignore_index, 0, 1)
                labels = labels * mask

                loss = self.loss_fn(logits, labels.float())

                loss = loss * mask
                loss = loss.mean()

        return logits, probs, loss


def determine_objective(num_classes: Union[int, List[int]]) -> Tuple[str, int, int]:
    if isinstance(num_classes, int):
        if num_classes == 2:
            return "binary", 2, 1

        elif num_classes > 2:
            return "multi-class", num_classes, 1

        else:
            ValueError(f"num_classes {num_classes} not supported")
    elif isinstance(num_classes, list):
        if all(c == 2 for c in num_classes):
            return "binary-multi-target", 2, len(num_classes)

        else:
            raise AttributeError("multi class (non binary), multi target objective not supported yet")
    else:
        raise AttributeError(f"provided num classes type {type(num_classes)} not supported")


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: Union[int, float], num_training_steps, last_epoch=-1):
    """
    adapted from https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup

    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int` or :obj:`float`):
            The number of steps for the warmup phase. if provided as float must be between 0.0 and 1.0 => totalsteps * factor => num_training_steps
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    assert isinstance(num_warmup_steps, int) or (
            num_warmup_steps >= .0 and num_warmup_steps <= 1.0), "num_warmup_steps must be between 0.0 or 1.0 or an integer"
    num_warmup_steps = int(num_training_steps * num_warmup_steps) if isinstance(num_warmup_steps,
                                                                                float) else num_warmup_steps

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_exponential_decay_scheduler(optimizer, decay_rate: float, decay_step: int, last_epoch: int = -1):
    """

    applies a exponential decaying factor to the learning rate - decayed_lr = lr * (decay_rate ^ (current_step / decay_step)

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        decay_rate (:obj:`float`):
            decaying rate
        decay_step (:obj:`int`):
            decaying steps
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Returns:

    """

    def lr_lambda(current_step: int):
        return decay_rate ** (current_step / decay_step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def time_it(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        print(f"function {fn.__name__} started")

        start = timer()
        output = fn(*args, **kwargs)
        runtime = timer() - start

        print(f"function {fn.__name__} ended - took {runtime:2f} seconds")

        return output

    return wrap