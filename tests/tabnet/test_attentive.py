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

    @pytest.mark.parametrize("batch_size, input_size, attentive_size, gamma, alpha, relaxation_type, attentive_type, kwargs",
                             [
                                 (10, 6, 12, 1.0, 2.0, "gamma_fixed", "sparsemax", {}),
                                 (10, 6, 12, 2.0, 2.0, "gamma_fixed", "sparsemax", {}),
                                 (10, 6, 12, 0.5, 2.0, "gamma_fixed", "sparsemax", {}),
                                 (1, 6, 12, 1.0, 2.0, "gamma_fixed", "sparsemax", {}),
                                 (10, 6, 12, 0.5, 2.0, "gamma_fixed", "sparsemax", {"bias": True, "momentum": 0.001}),
                                 (1, 6, 12, 1.0, 2.0, "gamma_fixed", "entmax_sparsemax", {}),
                                 (1, 6, 12, 1.0, 2.0, "gamma_fixed", "entmax_bisect", {}),
                                 (1, 6, 12, 1.0, 2.0, "gamma_fixed", "entmax_entmax15", {}),
                                 (1, 6, 12, 1.0, 2.0, "gamma_fixed", "softmax", {}),
                                 (1, 6, 12, 1.0, 2.0, "gamma_trainable", "softmax", {}),
                                 (1, 6, 12, 1.0, torch.nn.Parameter(torch.scalar_tensor(2.0), requires_grad=True), "gamma_fixed", "alpha_trainable", {}),
                                 (1, 6, 12, 1.0, torch.nn.Parameter(torch.scalar_tensor(2.0), requires_grad=True), "gamma_trainable", "alpha_trainable", {}),
                             ])
    def test_autograd(self, batch_size, input_size, attentive_size, gamma, alpha, relaxation_type, attentive_type, kwargs):
        """autograd attentive transformer test"""
        from tabnet.attentive import AttentiveTransformer

        torch.random.manual_seed(1)
        feature = torch.randn(size=(batch_size, attentive_size), dtype=torch.double, requires_grad=True)
        prior = torch.ones(size=(batch_size, input_size), dtype=torch.double, requires_grad=True)

        at = AttentiveTransformer(attentive_size=attentive_size,
                                  input_size=input_size,
                                  gamma=gamma,
                                  alpha=alpha,
                                  relaxation_type=relaxation_type,
                                  attentive_type=attentive_type,
                                  **kwargs)
        at.double()

        assert torch.autograd.gradcheck(at, [feature, prior])
