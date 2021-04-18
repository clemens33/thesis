import math
from typing import Optional, List

import torch
import torch.nn as nn

from tabnet.utils import GhostBatchNorm1d


class FeatureLayer(nn.Module):
    def __init__(self, input_size: int, feature_size: int, bias: bool = False, **kwargs):
        super(FeatureLayer, self).__init__()

        self.feature_size = feature_size
        self.fc = nn.Linear(in_features=input_size, out_features=feature_size * 2, bias=bias)
        self.bn = GhostBatchNorm1d(input_size=feature_size * 2, **kwargs)

        self.glu = lambda input, n_units: input[..., :n_units] * torch.sigmoid(input[..., n_units:])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feature = self.fc(input)
        feature = self.bn(feature)

        # output = self.bn(self.fc(input))
        feature = self.glu(feature, self.feature_size)

        return feature


class FeatureTransformer(nn.Module):
    def __init__(self, input_size: int, feature_size: int, nr_layers: int = 1, shared_layers: Optional[List[FeatureLayer]] = None,
                 normalize: float = math.sqrt(.5), **kwargs):
        super(FeatureTransformer, self).__init__()

        assert len(shared_layers) + nr_layers > 0, f"there must be at least one layer"

        self.normalize = normalize

        if shared_layers:
            self.layers = nn.ModuleList(
                shared_layers +
                [FeatureLayer(input_size=feature_size, feature_size=feature_size, **kwargs) for _ in range(nr_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [FeatureLayer(input_size=input_size, feature_size=feature_size, **kwargs)] +
                [FeatureLayer(input_size=feature_size, feature_size=feature_size, **kwargs) for _ in range(nr_layers - 1)]
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feature = input
        for layer_nr, layer in enumerate(self.layers):
            feature = (layer(feature) + feature) * self.normalize if layer_nr > 0 else layer(feature)

        return feature
