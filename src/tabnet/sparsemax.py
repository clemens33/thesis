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


class _Sparsemax2(torch.autograd.Function):
    """gradient checking fails!!!! TODO check"""
    """adapted from https://github.com/Qwicen/node/blob/master/lib/nn_utils.py"""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int = -1) -> torch.Tensor:  # noqa
        ctx.dim = dim

        x -= x.max(dim=dim, keepdim=True)[0]  # for numerical stability

        #
        x_sorted, _ = torch.sort(x, descending=True, dim=dim)
        x_cumsum = x_sorted.cumsum(dim=dim) - 1  # cumulative summation

        #
        d = x.size(dim)
        rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
        view = [1] * x.dim()
        view[0] = -1
        rhos = rho.view(view)
        rhos = rhos.transpose(0, dim)
        #

        support = rhos * x_sorted > x_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = x_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(x.dtype)
        #

        output = torch.clamp(x - tau, min=0)
        ctx.save_for_backward(support_size, output)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # noqa
        ddim = None

        support_size, output = ctx.saved_tensors
        dim = ctx.dim
        dx = grad_output.clone()
        dx[output == 0] = 0

        v_hat = dx.sum(dim=dim) / support_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        dx = torch.where(output != 0, dx - v_hat, dx)
        return dx, ddim


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        super(Sparsemax, self).__init__()

        self.dim = dim
        self.sparsemax = _Sparsemax.apply

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sparsemax(input, self.dim)


# basic tests
if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_printoptions(precision=20)

    B = 4
    T = 3
    C = 20

    input = torch.randn(size=(B, T, C), dtype=torch.double, requires_grad=True)
    s = Sparsemax()

    ###

    output = s(input)
    ones = torch.sum(output, dim=-1)

    assert torch.allclose(ones, torch.ones_like(ones))

    print(f"numerical issues: {ones}")  # TODO check numerical issues maybe?
    # assert (ones == 1).all(), f"last dimension does not sum up to 1"
    ###

    non_zeros = torch.count_nonzero(output, dim=-1)
    print(f"sparsity is {1 - non_zeros / C}")

    ###

    test = torch.autograd.gradcheck(s, input)
    print(f"gradient checking: {test}")

    ###
