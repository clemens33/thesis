from typing import Tuple, Union

import torch
import torch.nn as nn
from entmax import Entmax15, Sparsemax

from tabnet.sparsemax import EntmaxBisect, Sparsemax as MySparsemax
from tabnet.utils import GhostBatchNorm1d


class Attentive(nn.Module):
    def __init__(self, dim: int = -1, attentive_type: str = "sparsemax", alpha: Union[nn.Parameter, float] = 2.0):
        super(Attentive, self).__init__()

        assert alpha >= 1.0, f"alpha {alpha} must be greater or equal than 1.0"

        self.attentive_type = attentive_type

        if isinstance(alpha, nn.Parameter):
            self.alpha = alpha
            self.attentive = EntmaxBisect(alpha=self.alpha, dim=dim)
        else:
            self.alpha = nn.Parameter(torch.scalar_tensor(alpha), requires_grad=(attentive_type == "alpha_trainable"))

            if self.attentive_type == "alpha_trainable":
                self.attentive = EntmaxBisect(alpha=self.alpha, dim=dim)
            elif self.attentive_type == "entmax_sparsemax":
                self.attentive = Sparsemax(dim=dim)
            elif self.attentive_type == "sparsemax":
                self.attentive = MySparsemax(dim=dim)
            elif self.attentive_type == "softmax":
                self.attentive = nn.Softmax(dim=dim)
            elif self.attentive_type == "entmax_entmax15":
                self.attentive = Entmax15(dim=dim)
            elif self.attentive_type == "entmax_bisect":
                self.attentive = EntmaxBisect(alpha=self.alpha, dim=dim)
            else:
                raise ValueError(f"attentive type {attentive_type} is unknown")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.attentive(input)


class AttentiveTransformer(nn.Module):
    def __init__(self,
                 attentive_size: int,
                 input_size: int,
                 gamma: Union[nn.Parameter, float] = 1.0,
                 relaxation_type: str = "gamma_fixed",
                 alpha: Union[nn.Parameter, float] = 2.0,
                 attentive_type: str = "sparsemax",
                 bias: bool = False,
                 **kwargs):
        super(AttentiveTransformer, self).__init__()

        gamma_trainable = (relaxation_type == "gamma_trainable")
        self.gamma = nn.Parameter(torch.scalar_tensor(gamma), requires_grad=gamma_trainable) if isinstance(gamma, float) else gamma

        self.fc = nn.Linear(in_features=attentive_size, out_features=input_size, bias=bias)
        self.bn = GhostBatchNorm1d(input_size=input_size, **kwargs)

        self.attentive = Attentive(dim=-1, alpha=alpha, attentive_type=attentive_type)

    def forward(self, feature: torch.Tensor, prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.bn(self.fc(feature))

        input = input * prior
        mask = self.attentive(input)

        # TODO check what if gamma is an integer
        prior = prior * (self.gamma - mask)

        return mask, prior
