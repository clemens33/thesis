from typing import Union, List

import torch
from torch.optim.lr_scheduler import LambdaLR

import torch.nn as nn


class MultiEmbedding(nn.Module):
    def __init__(self,
                 embedding_indices: List[int],
                 num_embeddings: List[int],
                 embedding_dims: List[int],
                 ):
        super(MultiEmbedding, self).__init__()

        assert len(num_embeddings) == len(
            embedding_dims), f"num_embeddings length {len(num_embeddings)} must be the same as embedding_dims length {len(embedding_dims)}"

        self.embedding_indices = embedding_indices

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_dim, embedding_dim=dim) for num_dim, dim in zip(num_embeddings, embedding_dims)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for idx, embedding in zip(self.embedding_indices, self.embeddings):
            input = inputs[..., idx].long()

            output = embedding(input)
            inputs[..., idx] = output[..., 0]

        return inputs


class StackedEmbedding(nn.Module):
    def __init__(self,
                 embedding_indices: List[int],
                 num_embeddings: List[int],
                 embedding_dim: int = 1,
                 stack: bool = True,
                 ):
        super(StackedEmbedding, self).__init__()

        assert embedding_dim == 1, f"only 1 is supported at the moment for embedding_dim"
        assert len(embedding_indices) == len(
            num_embeddings), f"length of embedding_indices {len(embedding_indices)} do not match length of num_embeddings {len(num_embeddings)}"

        self.stack = stack

        self.register_buffer("embedding_indices", torch.LongTensor(embedding_indices))
        self.register_buffer("offsets", torch.LongTensor([
            sum(num_embeddings[:i]) for i in range(len(num_embeddings))
        ]))

        self.embeddings = nn.Embedding(sum(num_embeddings), embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        _input = input[..., self.embedding_indices]
        _input = _input + self.offsets

        output = self.embeddings(_input.long())

        if self.stack:
            output = output.reshape(*input.size()[:-1], -1)

        input[..., self.embedding_indices] = output

        return input


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
