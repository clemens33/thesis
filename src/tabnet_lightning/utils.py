import tempfile
import uuid
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect


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
        labels = [str(l.item()) if l.ndim == 0 else str(l).replace("tensor(", "").replace(")", "") for l in labels]
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