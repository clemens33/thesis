from typing import Tuple, Optional, List, Union

import torch
import torch.nn as nn

from tabnet.feature import FeatureTransformer, FeatureLayer
from tabnet.step import Step
from tabnet.utils import GhostBatchNorm1d


class TabNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 decision_size: int,
                 nr_layers: int = 1,
                 nr_shared_layers: int = 1,
                 nr_steps: int = 1,
                 gamma: float = 1.0,
                 eps: float = 1e-5,
                 momentum: float = 0.01,
                 normalize_input: bool = True,
                 #
                 attentive_type: str = "sparsemax",
                 alpha: float = 2.0,
                 relaxation_type: str = "gamma_fixed",
                 #
                 return_all: bool = True,
                 **kwargs
                 ):
        super(TabNet, self).__init__()

        assert nr_steps > 0, "there must be at least one decision step"

        self.return_all = return_all
        self.eps = eps

        shared_layers = []
        if nr_shared_layers > 0:
            shared_layers.append(
                FeatureLayer.init_layer(input_size=input_size, feature_size=feature_size, **kwargs))
            shared_layers += [
                FeatureLayer.init_layer(input_size=feature_size, feature_size=feature_size, **kwargs)
                for _ in range(1, nr_shared_layers)]

        self.bn_input = GhostBatchNorm1d(input_size=input_size, momentum=momentum,
                                         virtual_batch_size=torch.iinfo(int).max) if normalize_input else None

        self.feature_transformer = FeatureTransformer(input_size=input_size, feature_size=feature_size, nr_layers=nr_layers,
                                                      shared_layers=shared_layers, momentum=momentum, **kwargs)

        gamma_shared_trainable = (relaxation_type == "gamma_shared_trainable")
        gamma = nn.Parameter(torch.scalar_tensor(gamma), requires_grad=gamma_shared_trainable) if gamma_shared_trainable else gamma

        alpha_shared_trainable = (attentive_type == "alpha_shared_trainable")
        alpha = nn.Parameter(torch.scalar_tensor(alpha), requires_grad=alpha_shared_trainable) if alpha_shared_trainable else alpha

        self.steps = nn.ModuleList([Step(input_size=input_size,
                                         feature_size=feature_size,
                                         decision_size=decision_size,
                                         nr_layers=nr_layers,
                                         shared_layers=shared_layers,
                                         gamma=gamma,
                                         relaxation_type=relaxation_type,
                                         alpha=alpha,
                                         attentive_type=attentive_type,
                                         momentum=momentum,
                                         **kwargs) for _ in range(nr_steps)])

    def forward(self, input: torch.Tensor, prior: Optional[torch.Tensor] = None) -> Union[Tuple[
                                                                                              torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                              List[torch.Tensor], List[torch.Tensor]],
                                                                                          Tuple[
                                                                                              torch.Tensor, torch.Tensor, torch.Tensor]]:

        input = self.bn_input(input) if self.bn_input is not None else input
        feature = self.feature_transformer(input)

        decisions, masks = [], []
        decisions_aggregated, masks_aggregated, entropy_aggregated = None, None, None

        prior = torch.ones_like(input) if prior is None else prior
        for step in self.steps:
            decision, feature, mask, prior = step(input, feature, prior)

            # store individual decisions and masks
            decisions.append(decision)
            masks.append(mask)

            # aggregate decisions
            decisions_aggregated = decision if decisions_aggregated is None else decisions_aggregated + decision

            # calculate aggregated mask
            scale_mask = torch.sum(decision, dim=-1, keepdim=True) / len(self.steps)
            masks_aggregated = mask * scale_mask if masks_aggregated is None else masks_aggregated + mask * scale_mask

            # calculate total entropy
            # _entropy = torch.mean(torch.sum(mask * torch.log(mask + self.eps), dim=-1), dim=-1)
            _entropy = torch.mean(torch.sum(-mask * torch.log(mask + self.eps), dim=-1), dim=-1) / len(self.steps)
            entropy_aggregated = _entropy if entropy_aggregated is None else entropy_aggregated + _entropy

        # entropy_aggregated /= len(self.steps)

        if self.return_all:
            return decisions_aggregated, masks_aggregated, entropy_aggregated, decisions, masks
        else:
            return decisions_aggregated, masks_aggregated, entropy_aggregated
