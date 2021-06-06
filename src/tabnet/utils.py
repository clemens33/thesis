from typing import List

import torch
import torch.nn as nn


class GhostBatchNorm1d(nn.Module):
    def __init__(self, input_size: int, momentum: float = 0.1, virtual_batch_size: int = 8, **kwargs):
        super(GhostBatchNorm1d, self).__init__()

        self.input_size = input_size
        self.virtual_batch_size = virtual_batch_size

        self.bn = nn.BatchNorm1d(self.input_size, momentum=momentum)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies ghost batch norm over 2D and 3D input. For 3D input ghost batch norm is applied over the batch size and sequence length.

        Args:
            input (torch.Tensor): must be of size (batch_size, input_size) or (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor of the same size as input

        """
        # skip for batch size 1 and training
        if (len(input) == 1 and self.training) or self.virtual_batch_size == -1:
            return input

        # resize to (batch_size, input_size, sequence_length) - the pytorch default batch norm dimensions for 3 dimensional data
        input = input.transpose(-1, 1) if input.ndim == 3 else input

        # vb is set to 1 in the original tabnet tf implementation during evaluation - TODO check why?
        vbs = self.virtual_batch_size
        # vbs = self.virtual_batch_size if self.training else 1

        # apply batch norm per chunk
        chunks = torch.split(input, vbs, dim=0)
        output = [self.bn(chunk) for chunk in chunks]
        output = torch.cat(output, dim=0)

        #  resize back if necessary
        output = output.transpose(-1, 1) if input.ndim == 3 else output

        return output


class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, th: float) -> torch.Tensor: # noqa
        # x[x >= th] = 1
        # x[x < th] = 0

        # x = torch.where(
        #     x >= th,
        #     torch.scalar_tensor(1, device=x.device),
        #     torch.scalar_tensor(0, device=x.device)
        # )

        # fastest version
        x = (x > th).float()

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): # noqa
        dth = None  # indeterminable
        dx = grad_output  # identity/pass through gradient

        return dx, dth


class Round1(nn.Module):
    def __init__(self, th: float = 0.5):
        super(Round1, self).__init__()

        self.th = nn.Parameter(torch.scalar_tensor(th), requires_grad=False)
        self.round = _Round.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.round(x, self.th)




class HardSigm1(nn.Module):
    def __init__(self, slope: float = 1.0, slope_type: str = "slope_fixed"):
        super(HardSigm1, self).__init__()

        slope_trainable = (slope_type == "slope_trainable")
        self.slope = nn.Parameter(torch.scalar_tensor(slope), requires_grad=slope_trainable)

        self.one = nn.Parameter(torch.scalar_tensor(1), requires_grad=False)
        self.zero = nn.Parameter(torch.scalar_tensor(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min((self.slope * x + 1) / 2, self.one), self.zero)


class HardSigm2(nn.Module):
    """fastest version"""

    def __init__(self, slope: float = 1.0, slope_type: str = "slope_fixed"):
        super(HardSigm2, self).__init__()

        slope_trainable = (slope_type == "slope_trainable")
        self.slope = nn.Parameter(torch.scalar_tensor(slope), requires_grad=slope_trainable)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = 0.5 * (self.slope * x + 1)

        return torch.clamp(t, min=0, max=1)


class SlopeScheduler():
    _name = "hardsigm.slope"

    def __init__(self, params, factor: int = 25, max: int = 5) -> None:
        super(SlopeScheduler, self).__init__()

        self.params = params

        # a = min(5, a + factor * epoch)
        self.update = 1 / factor
        self.max = max

    def step(self) -> None:
        for name, p in self.params.items():
            # if not p.requires_grad and self._name in name:
            if self._name in name:
                v = p.data + self.update
                p.data = v if v <= self.max else torch.scalar_tensor(self.max)

    def set_slopes(self, slope: float):
        for name, p in self.params.items():
            if self._name in name:
                p = torch.scalar_tensor(slope)

    def get_slopes(self) -> List[float]:
        slopes = []
        for name, p in self.params.items():
            if self._name in name:
                slopes.append(float(p.data))

        return slopes

    def get_slope(self, idx: int = 0) -> float:
        return self.get_slopes()[idx]
