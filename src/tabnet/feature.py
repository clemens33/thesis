import math
from typing import Optional, List

import torch
import torch.nn as nn

from tabnet.utils import GhostBatchNorm1d


class FeatureLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = False, **kwargs):
        super(FeatureLayer, self).__init__()

        self.hidden_size = hidden_size
        self.fc = nn.Linear(in_features=input_size, out_features=hidden_size * 2, bias=bias)
        self.bn = GhostBatchNorm1d(input_size=hidden_size * 2, **kwargs)

        self.glu = lambda input, n_units: input[..., :n_units] * torch.sigmoid(input[..., n_units:])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.bn(self.fc(input))
        output = self.glu(output, self.hidden_size)

        return output


class FeatureTransformer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, nr_layers: int = 1, shared_layers: Optional[List[FeatureLayer]] = None,
                 normalize: float = math.sqrt(.5),
                 **kwargs):
        super(FeatureTransformer, self).__init__()

        self.normalize = normalize

        if shared_layers:
            self.layers = nn.ModuleList(
                shared_layers +
                [FeatureLayer(input_size=hidden_size, hidden_size=hidden_size, **kwargs) for _ in range(nr_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [FeatureLayer(input_size=input_size, hidden_size=hidden_size)] +
                [FeatureLayer(input_size=hidden_size, hidden_size=hidden_size, **kwargs) for _ in range(nr_layers - 1)]
            )

        assert len(self.layers) > 0, f"there must be at least one layer"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        for layer_nr, layer in enumerate(self.layers):
            output = layer(output) + output * self.normalize if layer_nr > 0 else layer(output)

        return output


# some basic tests
if __name__ == '__main__':
    B = 128  # batch_size
    D = 2048  # input size
    H = 32  # hidden size

    input = torch.randn(size=(B, D))

    ###

    fl = FeatureLayer(input_size=D, hidden_size=H)
    print(
        f"number of trainable parameters of one feature layer having D {D} dimensions is {sum(p.numel() for p in fl.parameters() if p.requires_grad)}")

    output = fl(input)

    print(output)

    ###

    shared_layers = [FeatureLayer(input_size=D, hidden_size=H), FeatureLayer(input_size=H, hidden_size=H)]
    ft = FeatureTransformer(nr_layers=2, shared_layers=shared_layers, input_size=D, hidden_size=H)
    print(
        f"number of trainable parameters of one feature transformer having D {D} dimensions and 4 layers is {sum(p.numel() for p in ft.parameters() if p.requires_grad)}")

    output = ft(input)

    print(output)

    ###

    # shared_layers = [FeatureLayer(input_size=D), FeatureLayer(input_size=D)]
    ft = FeatureTransformer(nr_layers=1, shared_layers=None, input_size=D, hidden_size=H)
    print(
        f"number of trainable parameters of one feature transformer having D {D} dimensions and 2 layers is {sum(p.numel() for p in ft.parameters() if p.requires_grad)}")

    output = ft(input)

    print(output)

    ###

    ft = FeatureTransformer(nr_layers=0, shared_layers=None, input_size=D, hidden_size=H)
    output = ft(input)

    print(output)

    ###

    ft = FeatureTransformer(nr_layers=4, shared_layers=None, input_size=D, hidden_size=H)

    output = ft(input)

    print(output)

    ###
