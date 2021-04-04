import pytest
import torch


class TestAttentiveTransformer():

    @pytest.mark.parametrize("batch_size, input_size, attentive_size, gamma, kwargs",
                             [
                                 (10, 6, 12, 1.0, {}),
                                 (10, 6, 12, 2.0, {}),
                                 (10, 6, 12, 0.5, {}),
                                 (1, 6, 12, 1.0, {}),
                                 (10, 6, 12, 0.5, {"bias": True, "momentum": 0.001}),
                             ])
    def test_at_forward(self, batch_size, input_size, attentive_size, gamma, kwargs):
        """test attentive transformer forward path and expected dimensions"""
        from tabnet.attentive import AttentiveTransformer

        feature = torch.randn(size=(batch_size, attentive_size))
        prior = torch.ones(size=(batch_size, input_size), dtype=feature.dtype)

        at = AttentiveTransformer(attentive_size=attentive_size, input_size=input_size, gamma=gamma, **kwargs)

        if batch_size == 1:
            at.eval()

        mask, prior = at(feature, prior)

        assert (mask + prior == gamma).all()
        assert mask.shape[0] == batch_size
        assert mask.shape[1] == input_size
