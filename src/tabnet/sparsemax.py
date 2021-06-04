from typing import Any, Tuple, Union

import torch
import torch.nn as nn
from entmax import entmax_bisect


class _Sparsemax1(torch.autograd.Function):
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
            ctx, input = _Sparsemax1._flatten_all_but_nth_dim(ctx, input)

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
            ctx, output = _Sparsemax1._unflatten_all_but_nth_dim(ctx, output)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # noqa
        output, *_ = ctx.saved_tensors

        # Reshape if needed
        if ctx.needs_reshaping:
            ctx, grad_output = _Sparsemax1._flatten_all_but_nth_dim(ctx, grad_output)

        # Compute gradient
        nonzeros = torch.ne(output, 0)
        num_nonzeros = nonzeros.sum(-1, keepdim=True)
        sum = (grad_output * nonzeros).sum(-1, keepdim=True) / num_nonzeros
        grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, grad_input = _Sparsemax1._unflatten_all_but_nth_dim(ctx, grad_input)

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
    # credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
    # TODO this version fails gradient checking - refer to tests - check why?
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):  # noqa
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _Sparsemax2._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _Sparsemax2._make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size

    @staticmethod
    def _make_ix_like(input, dim=0):
        d = input.size(dim)
        rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
        view = [1] * input.dim()
        view[0] = -1
        return rho.view(view).transpose(0, dim)


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        super(Sparsemax, self).__init__()

        self.dim = dim
        self.sparsemax = _Sparsemax1.apply

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sparsemax(input, self.dim)


class EntmaxBisect(nn.Module):
    def __init__(self, alpha: Union[nn.Parameter, float] = 1.5, dim: int = -1, n_iter: int = 50):
        super().__init__()

        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha

    def forward(self, X):
        return entmax_bisect(
            X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter
        )
