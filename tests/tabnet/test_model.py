import pytest
import torch
import logging

logger = logging.getLogger()


class TestTabNet():

    @pytest.mark.parametrize("input_dim, feature_size, decision_size, nr_layers, nr_shared_layers, nr_steps, gamma, kwargs",
                             [((10, 64), 32, 16, 2, 2, 3, 1.0, {}),
                              ((1, 64), 32, 16, 2, 2, 3, 1.0, {}),
                              # ((100, 10, 128), 32, 16, 2, 2, 3, 1.0, {}),
                              ])
    def test_tabnet_forward(self,
                            input_dim,
                            feature_size,
                            decision_size,
                            nr_layers,
                            nr_shared_layers,
                            nr_steps,
                            gamma,
                            kwargs):
        """test tabnet forward path and expected dimensions"""
        from tabnet import TabNet

        input = torch.randn(size=input_dim)

        batch_size = input.shape[0]
        input_size = input.shape[-1]

        tabnet = TabNet(input_size=input_size,
                        feature_size=feature_size,
                        decision_size=decision_size,
                        nr_layers=nr_layers,
                        nr_shared_layers=nr_shared_layers,
                        nr_steps=nr_steps,
                        gamma=gamma)

        if batch_size == 1:
            tabnet.eval()

        decision, mask, entropy = tabnet(input)

        assert decision.size() == (batch_size, decision_size)
        assert mask.size() == (batch_size, input_size)
        assert entropy > 0
