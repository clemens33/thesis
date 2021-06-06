from typing import Tuple, Union

import torch
import torch.nn as nn
from entmax import Entmax15, Sparsemax

from tabnet.sparsemax import EntmaxBisect, Sparsemax as MySparsemax
from tabnet.utils import GhostBatchNorm1d, Round1 as Round, HardSigm2 as HardSigm


class Attentive(nn.Module):
    def __init__(self, dim: int = -1, attentive_type: str = "sparsemax", alpha: Union[nn.Parameter, float] = 2.0, **kwargs):
        super(Attentive, self).__init__()

        assert alpha >= 1.0 or alpha == -2.0, f"alpha {alpha} must be greater or equal than 1.0"

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
            elif self.attentive_type == "binary_mask":
                self.attentive = _BinaryMask(**kwargs)
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

        assert relaxation_type not in [
            "gamma_shared_trainable, gamma_trainable, gamma_fixed"], f"relaxation type {relaxation_type} is unknown"

        gamma_trainable = (relaxation_type == "gamma_trainable")
        self.gamma = nn.Parameter(torch.scalar_tensor(gamma), requires_grad=gamma_trainable) if isinstance(gamma, float) else gamma

        self.fc = nn.Linear(in_features=attentive_size, out_features=input_size, bias=bias)
        self.bn = GhostBatchNorm1d(input_size=input_size, **kwargs)

        self.attentive = Attentive(dim=-1, alpha=alpha, attentive_type=attentive_type, **kwargs)

    def forward(self, feature: torch.Tensor, prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attentive_feature = self.bn(self.fc(feature))

        attentive_feature = attentive_feature * prior
        mask = self.attentive(attentive_feature)

        # TODO check what if gamma is an integer
        prior = prior * (self.gamma - mask)

        return mask, prior


class _BinaryMask(nn.Module):
    def __init__(self, slope: float = 1.0, slope_type: str = "slope_fixed", th: float = 0.5, **kwargs):
        super(_BinaryMask, self).__init__()

        self.round = Round(th)

        if slope_type == "slope_hard":
            self.activation = torch.nn.Hardsigmoid()
            self.activation.slope = slope
        else:
            self.activation = HardSigm(slope=slope, slope_type=slope_type)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activated = self.activation(inputs)
        #mask_tilde = torch.sigmoid(inputs)
        mask = self.round(activated)

        # mask = torch.sigmoid(sz)

        return mask
