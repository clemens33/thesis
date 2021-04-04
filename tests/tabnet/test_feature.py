import pytest
import torch


class TestFeatureLayer():

    @pytest.mark.parametrize("batch_size, input_size, feature_size, kwargs",
                             [
                                 (128, 2048, 32, {}),
                                 (1, 128, 64, {}),
                                 (10, 128, 64, {"bias": True, "momentum": 0.01, "virtual_batch_size": 2})
                             ])
    def test_fl_forward(self, batch_size, input_size, feature_size, kwargs):
        """test feature layer forward path and expected dimensions"""
        from tabnet.feature import FeatureLayer

        input = torch.randn(size=(batch_size, input_size))

        fl = FeatureLayer(input_size=input_size, feature_size=feature_size, **kwargs)

        output = fl(input)

        assert output.shape[0] == batch_size
        assert output.shape[1] == feature_size


class TestFeatureTransformer():

    @pytest.mark.parametrize("batch_size, input_size, feature_size, nr_layers, nr_shared_layers",
                             [
                                 (10, 128, 32, 1, 1),
                                 (10, 128, 32, 1, 0),
                                 (10, 128, 32, 0, 1),
                                 (10, 128, 32, 3, 3),
                                 (1, 128, 32, 2, 2),
                             ])
    def test_ft_forward(self, batch_size, input_size, feature_size, nr_layers, nr_shared_layers):
        """test feature transformer forward path and expected dimensions and number of layers"""
        from tabnet.feature import FeatureTransformer, FeatureLayer

        input = torch.randn(size=(batch_size, input_size))

        shared_layers = []
        if nr_shared_layers > 0:
            shared_layers.append(FeatureLayer(input_size=input_size, feature_size=feature_size))
            shared_layers += [FeatureLayer(input_size=feature_size, feature_size=feature_size) for _ in range(1, nr_shared_layers)]

        ft = FeatureTransformer(nr_layers=nr_layers, shared_layers=shared_layers, input_size=input_size, feature_size=feature_size)

        if batch_size == 1:
            ft.eval()

        output = ft(input)

        assert len(ft.layers) == nr_layers + nr_shared_layers
        assert output.shape[0] == batch_size
        assert output.shape[1] == feature_size
