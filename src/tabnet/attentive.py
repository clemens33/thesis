from typing import Tuple

import torch
import torch.nn as nn

from tabnet.sparsemax import Sparsemax
from tabnet.utils import GhostBatchNorm1d


class AttentiveTransformer(nn.Module):
    def __init__(self, input_size: int, gamma: float = 1.0, **kwargs):
        super(AttentiveTransformer, self).__init__()

        self.gamma = gamma

        self.fc = nn.Linear(in_features=input_size, out_features=input_size)
        self.bn = GhostBatchNorm1d(input_size=input_size, **kwargs)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, input: torch.Tensor, prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.bn(self.fc(input))

        mask = mask * prior
        mask = self.sparsemax(mask)

        prior = prior * (self.gamma - mask)

        return mask, prior


if __name__ == "__main__":
    pass
