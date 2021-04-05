from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from tabnet import TabNet


class TabNetClassifier(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 decision_size: int,
                 num_classes: int,
                 nr_layers: int = 1,
                 nr_shared_layers: int = 1,
                 nr_steps: int = 1,
                 gamma: float = 1.0,
                 eps: float = 1e-5,
                 momentum: float = 0.01,
                 virtual_batch_size: int = 8,
                 **kwargs
                 ):
        super(TabNetClassifier, self).__init__()

        self.encoder = TabNet(input_size=input_size,
                              feature_size=feature_size,
                              decision_size=decision_size,
                              nr_layers=nr_layers,
                              nr_shared_layers=nr_shared_layers,
                              nr_steps=nr_steps,
                              gamma=gamma,
                              eps=eps,
                              momentum=momentum,
                              virtual_batch_size=virtual_batch_size,
                              **kwargs)

        self.classifier = nn.Linear(in_features=decision_size, out_features=num_classes)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decision, mask, entropy = self.encoder(inputs)

        logits = self.classifier(decision)

        return logits, mask, entropy

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor):
        pass

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("TabNetClassifier")

        parser.add_argument("--input_size", type=int, help="input feature size/dimensions")
        parser.add_argument("--feature_size", type=int, help="feature/hidden size")
        parser.add_argument("--decision_size", type=int, help="decision/encoder size - output of tabnet before applying any specific task")
        parser.add_argument("--num_classes", type=int, help="output size/number of possible output classes")
        parser.add_argument("--nr_layers", type=int, default=1, help="number of independent layers per feature transformer")
        parser.add_argument("--nr_shared_layers", type=int, default=1, help="number of shared layers over all feature transformer")
        parser.add_argument("--nr_steps", type=int, default=1, help="number of tabnet steps")
        parser.add_argument("--gamma", type=float, default=1.0, help="gamma/relaxation parameter, larger values mean more relaxed "
                                                                     "behavior to reuse input features over subsequent steps")
        parser.add_argument("--eps", type=float, default=1e-5, help="for numerical stability calculating entropy used in regularization")
        parser.add_argument("--momentum", type=float, default=0.1, help="momentum for batch normalization")
        parser.add_argument("--virtual_batch_size", type=int, default=8, help="virtual batch size for ghost batch norm")

        return parent_parser
