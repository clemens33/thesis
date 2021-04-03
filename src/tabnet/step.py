from typing import Tuple, Optional, List

import torch
import torch.nn as nn

from tabnet.attentive import AttentiveTransformer
from tabnet.feature import FeatureTransformer, FeatureLayer


class Step(nn.Module):
    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 decision_size: int,
                 nr_layers: int = 1,
                 shared_layers: Optional[List[FeatureLayer]] = None,
                 gamma: float = 1.0,
                 decision_activation: nn.Module = nn.ReLU(),
                 **kwargs):
        super(Step, self).__init__()

        assert feature_size - decision_size > 0, "the size of features for calculation the attention transformer must be at least 1"

        self.decision_size = decision_size

        self.attentive_transformer = AttentiveTransformer(attentive_size=feature_size - decision_size, input_size=input_size, gamma=gamma,
                                                          **kwargs)
        self.feature_transformer = FeatureTransformer(input_size=input_size, feature_size=feature_size, nr_layers=nr_layers,
                                                      shared_layers=shared_layers, **kwargs)

        self.decision_activation = decision_activation

    def forward(self, input: torch.Tensor, feature: torch.Tensor, prior: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculation attentive mask
        mask, prior = self.attentive_transformer(feature[..., self.decision_size:], prior)

        # feature selection
        input = mask * input

        # calculate hidden features for decision output and for next step
        feature = self.feature_transformer(input)

        decision = feature[..., :self.decision_size]
        decision = self.decision_activation(decision)

        return decision, feature, mask, prior
