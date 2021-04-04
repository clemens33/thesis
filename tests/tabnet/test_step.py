import pytest
import torch


class TestStep():

    @pytest.mark.parametrize(
        "batch_size, input_size, feature_size, decision_size, nr_layers, nr_shared_layers, gamma, decision_activation, kwargs",
        [
            (10, 64, 32, 16, 2, 2, 1.0, torch.nn.ReLU(), {}),
            (1, 20, 18, 4, 1, 1, 1.0, torch.nn.ReLU(), {}),
            (4, 20, 18, 4, 1, 1, 1.0, torch.nn.ReLU(), {"bias": True}),
        ])
    def test_step_forward(self, batch_size, input_size, feature_size, decision_size, nr_layers, nr_shared_layers, gamma,
                          decision_activation, kwargs):
        """test decision step forward path and expected dimensions"""
        from tabnet.step import Step
        from tabnet.feature import FeatureLayer

        input = torch.randn(size=(batch_size, input_size))
        feature = torch.randn(size=(batch_size, feature_size))
        prior = torch.ones_like(input)

        shared_layers = []
        if nr_shared_layers > 0:
            shared_layers.append(FeatureLayer(input_size=input_size, feature_size=feature_size, **kwargs))
            shared_layers += [FeatureLayer(input_size=feature_size, feature_size=feature_size, **kwargs) for _ in
                              range(1, nr_shared_layers)]

        step = Step(input_size=input_size, feature_size=feature_size, decision_size=decision_size, nr_layers=nr_layers,
                    shared_layers=shared_layers, gamma=gamma, decision_activation=decision_activation, **kwargs)

        decision, feature, mask, prior = step(input, feature, prior)

        assert decision.size() == (batch_size, decision_size)
        assert feature.size() == (batch_size, feature_size)
        assert mask.size() == (batch_size, input_size)
        assert prior.size() == (batch_size, input_size)
