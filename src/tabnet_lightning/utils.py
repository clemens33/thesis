import tempfile
import uuid
from typing import Union, List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
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


def plot_masks(inputs: torch.Tensor,
               aggregated_mask: torch.Tensor,
               masks: Optional[List[torch.Tensor]] = None,
               feature_names: Optional[List[str]] = None,
               labels: Optional[torch.Tensor] = None,
               cmap: str = "viridis",
               # cmap: str = "binary",
               alpha: float = 1.0,
               out_fname: str = str(uuid.uuid4()),
               out_path: str = str(tempfile.mkdtemp()),
               show: bool = False,
               nr_samples: int = 20,
               normalize_inputs: bool = True,
               ) -> str:
    if masks is None:
        masks = []

    inputs = inputs.detach().cpu()
    normalized_inputs = (inputs.float() - inputs.float().mean(dim=0)) / inputs.float().std(dim=0)

    inputs = inputs[:nr_samples, ...]
    normalized_inputs = normalized_inputs[:nr_samples, ...]
    aggregated_mask = aggregated_mask.detach().cpu()[:nr_samples, ...]
    masks = [m.detach().cpu()[:nr_samples, ...] for m in masks] if masks is not None else None

    # cmap = plt.cm.get_cmap(cmap).reversed()
    plt.tight_layout()

    ratio = inputs.shape[0] / inputs.shape[1]
    offset = 3 if normalize_inputs else 2
    offset = offset + 1 if masks else offset
    nr_figures = offset + len(masks)

    w, h = figaspect(inputs) * 2  # * len(inputs) / 10
    fig, axes = plt.subplots(nrows=nr_figures,
                             ncols=1,
                             figsize=(w, h * nr_figures))
    axes = np.expand_dims(axes, axis=1) if len(axes.shape) == 1 else axes

    axes[0, 0].set_title("inputs")
    axes[0, 0].set_ylabel("sample #")
    pos = axes[0, 0].imshow(inputs, cmap=cmap, alpha=alpha, interpolation="none")
    fig.colorbar(pos, fraction=0.047 * ratio, ax=axes[0, 0])

    # only label x axis with features if under a defined nr of features
    if inputs.numel() < 2000:

        if feature_names:
            assert len(feature_names) == inputs.shape[
                -1], f"number of feature names {len(feature_names)} does not match the input features size {inputs.shape[-1]}"
            axes[0, 0].set_xticks(list(range(len(feature_names))))
            axes[0, 0].set_xticklabels(feature_names, rotation=45, ha="right", color="black", rotation_mode="anchor", fontsize=8)
            axes[0, 0].set_xlabel("feature")
        else:
            axes[0, 0].set_xticks(list(range(inputs.shape[-1])))
            axes[0, 0].set_xticklabels(list(range(inputs.shape[-1])), rotation=45, ha="right", color="black", rotation_mode="anchor",
                                       fontsize=8)
            axes[0, 0].set_xlabel("feature #")
    else:
        axes[0, 0].set_xlabel("feature #")

    # show normalized inputs
    if normalize_inputs:
        axes[1, 0].set_title("normalized inputs")
        axes[1, 0].set_ylabel("sample #")
        pos = axes[1, 0].imshow(normalized_inputs, cmap=cmap, alpha=alpha, interpolation="none")
        fig.colorbar(pos, fraction=0.047 * ratio, ax=axes[1, 0])

    # summed masks
    if masks:
        summed_mask = torch.stack(masks, dim=0).sum(dim=0)
        axes[offset - 2, 0].set_title("summed mask")
        axes[offset - 2, 0].set_ylabel("sample #")
        pos = axes[offset - 2, 0].imshow(summed_mask, cmap=cmap, alpha=alpha, interpolation="none")
        fig.colorbar(pos, fraction=0.047 * ratio, ax=axes[offset - 2, 0])
        axes[offset - 2, 0].set_xlabel("mask entry #")

    # aggregated mask
    axes[offset - 1, 0].set_title("aggregated mask")
    axes[offset - 1, 0].set_ylabel("sample #")
    pos = axes[offset - 1, 0].imshow(aggregated_mask, cmap=cmap, alpha=alpha, interpolation="none")
    fig.colorbar(pos, fraction=0.047 * ratio, ax=axes[offset - 1, 0])
    axes[offset - 1, 0].set_xlabel("mask entry #")

    # individual masks
    for i, m in enumerate(masks):
        axes[i + offset, 0].set_title("mask" + str(i))
        axes[i + offset, 0].set_ylabel("sample #")
        pos = axes[i + offset, 0].imshow(m, cmap=cmap, alpha=alpha, interpolation="none")
        fig.colorbar(pos, fraction=0.047 * ratio, ax=axes[i + offset, 0])
        axes[i + offset, 0].set_xlabel("mask entry #")

    # labels as y ticks
    if labels is not None:
        labels = labels.detach().cpu()[:nr_samples, ...]
        labels = [str(l.item()) for l in labels]
        # labels = labels.unsqueeze(dim=-1)

        for i in range(nr_figures):
            axes[i, 0].yaxis.tick_right()
            axes[i, 0].yaxis.set_label_position("right")
            axes[i, 0].set_ylabel("labels")
            axes[i, 0].set_yticks(list(range(len(labels))))
            axes[i, 0].set_yticklabels(labels, fontsize=8)

    if show:
        plt.show()

    path = out_path + "/" + out_fname + ".png"
    plt.savefig(path)

    return path


def plot_rankings(mask: torch.Tensor,
                  feature_names: Optional[List[str]] = None,
                  descending: bool = True,
                  show: bool = False,
                  plot_std: bool = True,
                  top_k: Optional[int] = None,
                  out_fname: str = str(uuid.uuid4()),
                  out_path: str = str(tempfile.mkdtemp())
                  ) -> str:
    feature_names = ["na" for i in range(mask.shape[-1])] if feature_names is None else feature_names
    feature_names = [f"{name}/{str(i)}" for i, name in enumerate(feature_names)]

    assert mask.shape[-1] == len(
        feature_names), f"number of feature names {len(feature_names)} must match feature dimensions {mask.shape[-1]}"

    mask = mask.detach().cpu()
    top_k = len(feature_names) if top_k is None else min(len(feature_names), top_k)

    fa_mean = torch.mean(mask, dim=0)
    fa_std = torch.std(mask, dim=0)

    sorted_fa_mean, indices = torch.sort(fa_mean, descending=not descending)
    sorted_fa_std = fa_std[indices]

    feature_names = np.array(feature_names)[indices]

    sorted_fa_mean = sorted_fa_mean[:top_k]
    sorted_fa_std = sorted_fa_std[:top_k]
    feature_names = feature_names[:top_k]

    # normalize
    s = sorted_fa_mean.sum()
    sorted_fa_mean = (sorted_fa_mean / s) * 100
    sorted_fa_std = (sorted_fa_std / s) * 100

    h = max(4, int(top_k / 2))
    w = max(6, int(h / 3))
    fig, ax = plt.subplots(figsize=(w, h))

    ax.barh(np.arange(len(feature_names)),
            sorted_fa_mean,
            xerr=sorted_fa_std if plot_std else None,
            color="tab:blue",
            alpha=0.6,
            error_kw={
                "elinewidth": 0.8,
                "alpha": 0.5
            })

    ax.set_ylabel("feature name / dimension")
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)

    for i, v in enumerate(sorted_fa_mean):
        ax.text(v, i, f"{v.item():.2f}%", va="bottom")

    ax.set_xlabel("mean +/- std feature importance in %")
    ax.set_title(f"feature importance/ranking - top {top_k}")

    plt.tight_layout()

    fig1 = plt.gcf()

    if show:
        plt.show()

    path = out_path + "/" + out_fname + ".png"
    fig1.savefig(path, transparent=False)

    return path


def replace_key_name(d: dict, to_replace: str, replace_with: str):
    return {
        k.replace(to_replace, replace_with) if isinstance(k, str) else k: v for k, v in d.items()
    }






# TODO - move to pytest
# if __name__ == "__main__":
#     # from datasets import CovTypeDataModule
#     #
#     # size = (100, 55)
#     # nr_samples = 20
#     #
#     # inputs = torch.randint(0, 2, size=size)
#     # labels = torch.randint(0, 8, size=(size[0],))
#     # mask = torch.rand(size)
#     #
#     # masks = [torch.rand(size) for _ in range(4)]
#     #
#
#     # plot_masks(inputs, mask, masks, feature_names=CovTypeDataModule.ALL_COLUMNS)
#     # print(plot_masks(inputs, nr_samples=nr_samples, aggregated_mask=mask, labels=None, normalize_inputs=False, show=True))
#
#     # print(plot_masks(inputs, nr_samples=nr_samples, aggregated_mask=mask, labels=labels, masks=masks, normalize_inputs=False, show=True))
#     # print(plot_masks(inputs, nr_samples=nr_samples, aggregated_mask=mask, labels=labels, masks=masks, normalize_inputs=True, show=True))
#     #
#     # print(plot_masks(inputs, nr_samples=nr_samples, aggregated_mask=mask, labels=labels, normalize_inputs=False, show=True))
#     # print(plot_masks(inputs, nr_samples=nr_samples, aggregated_mask=mask, labels=labels, normalize_inputs=True, show=True))
#
#     from datasets import CovTypeDataModule
#
#     size = (100, 10)
#
#     mask = torch.randint(0, 10, size=size).float()
#     # feature_names = CovTypeDataModule.ALL_COLUMNS[:10]
#
#     plot_rankings(mask, show=True, plot_std=False, top_k=100)
#
#     feature_names = CovTypeDataModule.ALL_COLUMNS[:5]
#     mask = torch.Tensor([
#         [10, 0, 0, 0, 1],
#         [11, 0, 0, 3, 0],
#         [11, 0, 0, 0, 1],
#         [12, 0, 0, 1, 0],
#         [10, 0, 0, 10, 2],
#     ])
#
#     path = plot_rankings(mask, feature_names=feature_names, show=True, plot_std=True, top_k=100)
#     print(path)
