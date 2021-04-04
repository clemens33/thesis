import logging

import pytest
import torch

logger = logging.getLogger()


class TestTabNet():

    @pytest.mark.parametrize("input_dim, feature_size, decision_size, nr_layers, nr_shared_layers, nr_steps, gamma, kwargs",
                             [
                                 ((10, 64), 32, 16, 2, 2, 3, 1.0, {}),
                                 ((1, 64), 32, 16, 2, 2, 3, 1.0, {}),
                                 ((100, 10, 128), 32, 16, 2, 2, 3, 1.0, {}),
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
        input_size = input.shape[-1]

        tabnet = TabNet(input_size=input_size,
                        feature_size=feature_size,
                        decision_size=decision_size,
                        nr_layers=nr_layers,
                        nr_shared_layers=nr_shared_layers,
                        nr_steps=nr_steps,
                        gamma=gamma)

        decision, mask, entropy = tabnet(input)

        expected_mask_dim = input_dim
        expected_decision_dim = input_dim[:-1] + (decision_size,)

        assert mask.size() == expected_mask_dim
        assert decision.size() == expected_decision_dim
        assert (entropy > torch.zeros_like(entropy)).all()

    @pytest.mark.parametrize("input_dim, feature_size, decision_size, nr_layers, nr_shared_layers, nr_steps, gamma, kwargs",
                             [
                                 ((1, 4), 10, 5, 2, 2, 3, 1.0, {}),
                                 ((2, 8), 10, 5, 2, 2, 3, 2.0, {}),
                                 ((2, 8), 10, 5, 1, 1, 3, 2.0, {}),
                                 ((2, 8), 10, 5, 0, 1, 3, 2.0, {}),
                                 ((2, 8), 10, 5, 1, 0, 3, 2.0, {}),
                             ])
    def test_shared_layers(self,
                           input_dim,
                           feature_size,
                           decision_size,
                           nr_layers,
                           nr_shared_layers,
                           nr_steps,
                           gamma,
                           kwargs):
        """test tabnet autograd"""
        from tabnet import TabNet
        from itertools import product

        input = torch.randn(size=input_dim)
        input_size = input.shape[-1]

        tabnet = TabNet(input_size=input_size,
                        feature_size=feature_size,
                        decision_size=decision_size,
                        nr_layers=nr_layers,
                        nr_shared_layers=nr_shared_layers,
                        nr_steps=nr_steps,
                        gamma=gamma)

        def _compare_modules(module1, module2):
            for key_item1, key_item2 in zip(module1.state_dict().items(), module2.state_dict().items()):
                if not torch.eq(key_item1[1], key_item1[1]).all():
                    return False

            return True

        # check shared layers
        shared_layers = tabnet.feature_transformer.layers[:nr_shared_layers]
        for step in tabnet.steps:
            for l in range(nr_shared_layers):
                s1 = shared_layers[l]
                s2 = step.feature_transformer.layers[l]

                assert _compare_modules(s1, s2)

        assert True

    @pytest.mark.parametrize("input_dim, feature_size, decision_size, nr_layers, nr_shared_layers, nr_steps, gamma, kwargs",
                             [
                                 ((1, 4), 10, 5, 2, 2, 3, 1.0, {}),
                                 ((2, 8), 10, 5, 2, 2, 3, 2.0, {}),
                                 ((1, 6, 4), 10, 5, 2, 2, 3, 1.0, {}),
                                 ((2, 6, 4), 10, 5, 2, 2, 3, 1.0, {}),
                             ])
    def test_autograd(self,
                      input_dim,
                      feature_size,
                      decision_size,
                      nr_layers,
                      nr_shared_layers,
                      nr_steps,
                      gamma,
                      kwargs):
        """test tabnet autograd"""
        from tabnet import TabNet

        input = torch.randn(size=input_dim, dtype=torch.double, requires_grad=True)
        input_size = input.shape[-1]

        tabnet = TabNet(input_size=input_size,
                        feature_size=feature_size,
                        decision_size=decision_size,
                        nr_layers=nr_layers,
                        nr_shared_layers=nr_shared_layers,
                        nr_steps=nr_steps,
                        gamma=gamma)

        # gradcheck by default works with double
        tabnet.double()

        assert torch.autograd.gradcheck(tabnet, input)
