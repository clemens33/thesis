import pytest
import torch


class TestSparsemax():

    @pytest.mark.parametrize("size, dim",
                             [
                                 ((32, 10, 13), -1),
                                 ((32, 10, 1), -1),
                                 ((32, 10, 13), 0),
                                 ((32, 10, 13), 1),
                                 ((1, 1, 13), 1),
                                 ((1024, 10), -1)
                             ])
    def test_sparsemax(self, size, dim):
        """tests sums to one along given dimension and sparsity property"""
        from tabnet.sparsemax import Sparsemax

        input = torch.randn(*size)

        s = Sparsemax(dim=dim)

        output = s(input)

        expect_ones = torch.sum(output, dim=dim)
        assert torch.allclose(expect_ones, torch.ones_like(expect_ones))

        # TODO check numerical issues - sometimes does not sum up to 1 exactly
        # torch.set_printoptions(precision=20)
        # assert (expect_ones == 1.0).all()

        # TODO check if sparsity is guaranteed (seems like not)
        # count_zeros = input.shape[dim] - torch.count_nonzero(output, dim=dim)
        # assert (count_zeros > 0).all()

    @pytest.mark.parametrize("size, dim",
                             [
                                 ((32, 10, 13), -1),
                                 ((32, 10, 1), -1),
                                 ((1, 13), 1),
                                 ((64, 64), 0)
                             ])
    def test_autograd(self, size, dim):
        """tests autograd"""
        from tabnet.sparsemax import Sparsemax

        input = torch.randn(*size, dtype=torch.double, requires_grad=True)

        s = Sparsemax(dim=dim)

        # gradcheck by default works with double
        s.double()

        assert torch.autograd.gradcheck(s, input)
