import pytest
import torch


class TestGhostBatchNorm1d():

    @pytest.mark.parametrize("batch_size, input_size, momentum, virtual_batch_size",
                             [(128, 32, 0.1, 4),
                              (1024, 512, 0.01, 128),
                              ])
    def test_statistics(self, batch_size, input_size, momentum, virtual_batch_size):
        """tests ghost batch norm statistics"""
        from tabnet.utils import GhostBatchNorm1d

        input = torch.randn(size=(batch_size, input_size))

        gbn = GhostBatchNorm1d(input_size=input_size, momentum=momentum, virtual_batch_size=virtual_batch_size)
        output = gbn(input)

        mean = torch.mean(output)
        std = torch.std(output)

        assert torch.allclose(mean, torch.tensor(0.0))
        assert torch.allclose(std, torch.tensor(1.0), atol=1e-4)
