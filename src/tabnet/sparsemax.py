from typing import Any, Tuple

import torch
import torch.nn as nn


class _Sparsemax(torch.autograd.Function):
    """adapted from https://github.com/aced125/sparsemax/tree/master/sparsemax"""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, dim: int = -1) -> torch.Tensor:  # noqa
        input_dim = input.dim()
        if input_dim <= dim or dim < -input_dim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{input_dim}, {input_dim - 1}], but got {dim})"
            )

        # Save operating dimension to context
        ctx.needs_reshaping = input_dim > 2
        ctx.dim = dim

        if ctx.needs_reshaping:
            ctx, input = _Sparsemax._flatten_all_but_nth_dim(ctx, input)

        # Translate by max for numerical stability
        input = input - input.max(-1, keepdim=True).values.expand_as(input)

        zs = input.sort(-1, descending=True).values
        range = torch.arange(1, input.size()[-1] + 1)
        range = range.expand_as(input).to(input)

        # Determine sparsity of projection
        bound = 1 + range * zs
        is_gt = bound.gt(zs.cumsum(-1)).type(input.dtype)
        k = (is_gt * range).max(-1, keepdim=True).values

        # Compute threshold
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (zs_sparse.sum(-1, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = torch.max(torch.zeros_like(input), input - taus)

        # Save context
        ctx.save_for_backward(output)

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, output = _Sparsemax._unflatten_all_but_nth_dim(ctx, output)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # noqa
        output, *_ = ctx.saved_tensors

        # Reshape if needed
        if ctx.needs_reshaping:
            ctx, grad_output = _Sparsemax._flatten_all_but_nth_dim(ctx, grad_output)

        # Compute gradient
        nonzeros = torch.ne(output, 0)
        num_nonzeros = nonzeros.sum(-1, keepdim=True)
        sum = (grad_output * nonzeros).sum(-1, keepdim=True) / num_nonzeros
        grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, grad_input = _Sparsemax._unflatten_all_but_nth_dim(ctx, grad_input)

        return grad_input, None

    @staticmethod
    def _flatten_all_but_nth_dim(ctx: Any, x: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """
        Flattens tensor in all but 1 chosen dimension.
        Saves necessary context for backward pass and unflattening.
        """

        # transpose batch and nth dim
        x = x.transpose(0, ctx.dim)

        # Get and save original size in context for backward pass
        original_size = x.size()
        ctx.original_size = original_size

        # Flatten all dimensions except nth dim
        x = x.reshape(x.size(0), -1)

        # Transpose flattened dimensions to 0th dim, nth dim to last dim
        return ctx, x.transpose(0, -1)

    @staticmethod
    def _unflatten_all_but_nth_dim(ctx: Any, x: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """
        Unflattens tensor using necessary context
        """
        # Tranpose flattened dim to last dim, nth dim to 0th dim
        x = x.transpose(0, 1)

        # Reshape to original size
        x = x.reshape(ctx.original_size)

        # Swap batch dim and nth dim
        return ctx, x.transpose(0, ctx.dim)


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        super(Sparsemax, self).__init__()

        self.dim = dim
        self.sparsemax = _Sparsemax.apply

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sparsemax(input, self.dim)
