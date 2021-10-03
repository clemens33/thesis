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


class Layer(nn.Module):
    def __init__(self, input_size: int, feature_size: int, bias: bool = False, **kwargs):
        super(Layer, self).__init__()

        self.input_size = input_size
        self.feature_size = feature_size

        self.fc = nn.Linear(in_features=input_size, out_features=feature_size, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.fc(input)


class FeatureLayer(nn.Module):
    """feature layer using gated linear unit activation"""

    def __init__(self, input_size: int, feature_size: int, dropout: float = 0.0, shared_layer: Optional[Layer] = None, **kwargs):
        super(FeatureLayer, self).__init__()

        if shared_layer is not None:
            assert shared_layer.input_size == input_size, f"shared_layer in_features {shared_layer.input_size} do not match input_size {input_size}"
            assert shared_layer.feature_size == feature_size * 2, f"shared_layer out_features {shared_layer.feature_size} do not match feature_size * 2 {feature_size}"

        self.fc = shared_layer if shared_layer is not None else FeatureLayer.init_layer(input_size=input_size, feature_size=feature_size,
                                                                                        **kwargs)
        self.bn = GhostBatchNorm1d(input_size=feature_size * 2, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.glu = GLU(n_units=feature_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feature = self.fc(input)
        feature = self.bn(feature)
        feature = self.dropout(feature) if self.dropout is not None else feature

        feature = self.glu(feature)

        return feature

    @staticmethod
    def init_layer(input_size: int, feature_size: int, **kwargs) -> Layer:
        """custom init function accounting for the double feature size due to the gated linear unit activation"""
        return Layer(input_size=input_size, feature_size=feature_size * 2, **kwargs)


class FeatureTransformer(nn.Module):
    def __init__(self, input_size: int, feature_size: int, nr_layers: int = 1, shared_layers: Optional[List[Layer]] = None,
                 normalize: float = math.sqrt(.5), **kwargs):
        super(FeatureTransformer, self).__init__()

        assert len(shared_layers) + nr_layers > 0, f"there must be at least one layer"

        self.normalize = normalize

        if shared_layers:
            self.layers = nn.ModuleList(
                [FeatureLayer(input_size=input_size, feature_size=feature_size, shared_layer=shared_layers[0], **kwargs)] +
                [FeatureLayer(input_size=feature_size, feature_size=feature_size, shared_layer=shared_layer, **kwargs) for shared_layer in
                 shared_layers[1:]] +
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
