import torch
import torch.nn as nn


class GhostBatchNorm1d(nn.Module):
    def __init__(self, input_size: int, virtual_batch_size: int = 8, **kwargs):
        super(GhostBatchNorm1d, self).__init__()

        self.input_size = input_size
        self.virtual_batch_size = virtual_batch_size

        self.bn = nn.BatchNorm1d(self.input_size, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(input, self.virtual_batch_size, dim=0)
        output = [self.bn(chunk) for chunk in chunks]

        return torch.cat(output, dim=0)


if __name__ == '__main__':
    B = 128
    D = 2048

    input = torch.randn(size=(B, D))

    ###

    gbn = GhostBatchNorm1d(input_size=D, virtual_batch_size=64)
    output = gbn(input)
    print(output)
