from typing import Tuple

import torch
import torch.nn as nn

from tabnet.sparsemax import Sparsemax
from tabnet.utils import GhostBatchNorm1d


class AttentiveTransformer(nn.Module):
    def __init__(self, attentive_size: int, input_size: int, gamma: float = 1.0, bias: bool = False, **kwargs):
        super(AttentiveTransformer, self).__init__()

        self.gamma = gamma

        self.fc = nn.Linear(in_features=attentive_size, out_features=input_size, bias=bias)
        self.bn = GhostBatchNorm1d(input_size=input_size, **kwargs)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, feature: torch.Tensor, prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.bn(self.fc(feature))

        input = input * prior
        mask = self.sparsemax(input)

        prior = prior * (self.gamma - mask)

        return mask, prior
