import torch
import torch.nn as nn


class GhostBatchNorm1d(nn.Module):
    def __init__(self, input_size: int, momentum: float = 0.1, virtual_batch_size: int = 8):
        super(GhostBatchNorm1d, self).__init__()

        self.input_size = input_size
        self.virtual_batch_size = virtual_batch_size

        self.bn = nn.BatchNorm1d(self.input_size, momentum=momentum)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(input, self.virtual_batch_size, dim=0)
        output = [self.bn(chunk) for chunk in chunks]

        return torch.cat(output, dim=0)
