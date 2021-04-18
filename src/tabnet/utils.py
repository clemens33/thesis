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
        if len(input) == 1 and self.training:
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
