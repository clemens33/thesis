from typing import Tuple, Optional

import torch
import torch.nn as nn

from tabnet.feature import FeatureTransformer, FeatureLayer
from tabnet.step import Step


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
                 momentum: float = 0.1,
                 **kwargs
                 ):
        super(TabNet, self).__init__()

        assert nr_steps > 0, "there must be at least one decision step"

        self.eps = eps

        shared_layers = []
        if nr_shared_layers > 0:
            shared_layers.append(
                FeatureLayer(input_size=input_size, feature_size=feature_size, momentum=momentum, **kwargs))
            shared_layers += [
                FeatureLayer(input_size=feature_size, feature_size=feature_size, momentum=momentum, **kwargs)
                for _ in range(1, nr_shared_layers)]

        self.bn = nn.BatchNorm1d(num_features=input_size, momentum=momentum)
        self.feature_transformer = FeatureTransformer(input_size=input_size, feature_size=feature_size, nr_layers=nr_layers,
                                                      shared_layers=shared_layers, momentum=momentum, **kwargs)

        self.steps = nn.ModuleList([Step(input_size=input_size, feature_size=feature_size, decision_size=decision_size, nr_layers=nr_layers,
                                         shared_layers=shared_layers, gamma=gamma, momentum=momentum, **kwargs) for _ in range(nr_steps)])

    def forward(self, input: torch.Tensor, prior: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = self.bn(input)
        feature = self.feature_transformer(input)

        decisions_aggregated, masks_aggregated, entropy_aggregated = None, None, None
        prior = torch.ones_like(input) if prior is None else prior
        for step in self.steps:
            decision, feature, mask, prior = step(input, feature, prior)

            # aggregate decisions
            decisions_aggregated = decision if decisions_aggregated is None else decisions_aggregated + decision

            # calculate aggregated mask
            scale_mask = torch.sum(decision, dim=-1, keepdim=True) / len(self.steps)
            masks_aggregated = mask * scale_mask if masks_aggregated is None else masks_aggregated + mask * scale_mask

            # calculate total entropy
            _entropy = torch.mean(torch.sum(-mask * torch.log(mask + self.eps), dim=-1), dim=-1) / len(self.steps)
            entropy_aggregated = _entropy if entropy_aggregated is None else entropy_aggregated + _entropy

        return decisions_aggregated, masks_aggregated, entropy_aggregated
