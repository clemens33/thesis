import torch
from torchmetrics import Metric


def sparsity(input: torch.Tensor) -> float:
    total = torch.numel(input)
    zeros = total - torch.count_nonzero(input)

    return zeros.float() / total


class Sparsity(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        self.add_state("zeros", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, input: torch.Tensor):  # noqa
        nonzeros = torch.count_nonzero(input)
        total = torch.numel(input)

        self.zeros += (total - nonzeros)
        self.total += total

    def compute(self):
        return self.zeros.float() / self.total
#
#
# if __name__ == "__main__":
#     inputs = torch.LongTensor([
#         [1, 1, 1, 0],
#         [0, 0, 0, 0],
#         [1, 1, 0, 0],
#     ])
#
#     print(sparsity(inputs))
#
#     metric = Sparsity()
#
#     metric.update(inputs)
#     metric.update(inputs)
#     metric.update(inputs)
#
#     s = metric.compute()
#
#     print(s)
#
#     inputs = torch.Tensor([
#         [0.1, 0.1, 0.1, .0],
#         [.0, .0, .0, .0],
#     ])
#
#     metric = Sparsity()
#
#     metric.update(inputs)
#     metric.update(inputs)
#     metric.update(inputs)
#
#     s = metric.compute()
#
#     print(s)
#
#     metric = Sparsity()
#
#     inputs = torch.LongTensor([
#         [1, 1, 1, 0],
#         [0, 0, 0, 0],
#         [1, 1, 0, 0],
#     ])
#
#     print(metric(inputs))
#
#     inputs = torch.Tensor([
#         [0.1, 0.1, 0.1, .0],
#         [.0, .0, .0, .0],
#     ])
#
#     print(metric(inputs))
#
#     print(metric.compute())
#
#
