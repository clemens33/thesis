import math
from typing import Optional, List

import torch
import torch.nn as nn

from tabnet.utils import GhostBatchNorm1d


class GLU(nn.Module):
    def __init__(self, n_units: int):
        super(GLU, self).__init__()

        self.n_units = n_units

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input[..., :self.n_units] * torch.sigmoid(input[..., self.n_units:])


class FeatureLayer(nn.Module):
    def __init__(self, input_size: int, feature_size: int, bias: bool = False, **kwargs):
        super(FeatureLayer, self).__init__()

        self.fc = nn.Linear(in_features=input_size, out_features=feature_size * 2, bias=bias)
        self.bn = GhostBatchNorm1d(input_size=feature_size * 2, **kwargs)

        self.glu = GLU(n_units=feature_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feature = self.fc(input)
        feature = self.bn(feature)

        feature = self.glu(feature)

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
