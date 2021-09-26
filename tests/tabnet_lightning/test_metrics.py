import pytest
import torch


class TestCustomAccuracy():
    @pytest.mark.parametrize("preds, targets, ignore_index, expected_value", [
        (torch.Tensor([0.1, 0.5, 0.6, 0.7, 0]), torch.Tensor([0, 1, 0, 0, -100]), -100, .5),

        (torch.Tensor([
            [0.1, 0.5, 0.6, 0.7],
        ]), torch.Tensor([
            [0, 1, 0, 0],
        ]), -100, .5),

        (torch.Tensor([
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ]), torch.Tensor([
            [0, 1, -100],
            [0, 1, -100],
            [0, 1, -100],
            [0, 1, -100],
        ]), -100, 0.5),

        (torch.Tensor([
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]), torch.Tensor([
            [0, 1, -100],
            [0, 1, 0],
            [0, 1, -100],
            [0, 1, 1],
        ]), -100, 0.666666),

        (torch.Tensor([
            [0.1, 1],
            [0.2, 1],
            [0.3, 1],
            [0.5, 1],
        ]), torch.Tensor([
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 1],
        ]), -100, 1.0),

    ])
    def test_accuracy(self, preds, targets, ignore_index, expected_value):
        from shared.metrics import CustomAccuracy

        num_targets = targets.shape[-1] if targets.ndim > 1 else 1
        fn = CustomAccuracy(num_targets, ignore_index, return_verbose=False)

        fn(preds, targets)
        value = fn.compute()

        assert torch.allclose(value, torch.tensor(expected_value))


class TestCustomAUROC():
    @pytest.mark.parametrize("preds, targets, ignore_index, chunks", [
        (torch.Tensor([
            [0.1, 0.3],
            [0.2, 0.4],
            [0.3, 0.6],
            [0.1, 0.1],
            [0.3, 0.6],
            [0.5, 0.1],
            [0.3, 0.6],
            [0.5, 0.1],
        ]), torch.Tensor([
            [0, 1],
            [0, 0],
            [0, -100],
            [1, 1],
            [0, -100],
            [1, 1],
            [0, -100],
            [1, 1],
        ]), -100, 4),

        (torch.Tensor([0.1, 0.9]), torch.Tensor([1, 0]), -100, 1),

        (torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), torch.Tensor([1, 0, 1, 0, 0, 0, 0, 0, 0]), -100, 9),

        (torch.Tensor([
            [0.1, 0.3],
            [0.2, 0.4],
            [0.3, 0.6],
            [0.1, 0.1],
        ]), torch.Tensor([
            [-100, 1],
            [-100, 0],
            [0, -100],
            [0, 1],
        ]), -100, 2),

        (torch.Tensor([
            [0.1, 0.3],
            [0.2, 0.4],
            [0.3, 0.6],
            [0.1, 0.1],
        ]), torch.Tensor([
            [-100, 1],
            [-100, 0],
            [-100, -100],
            [-100, 1],
        ]), -100, 2),

        (torch.Tensor([
            [0.1, 0.3],
            [0.2, 0.4],
            [0.3, 0.6],
            [0.1, 0.1],
            [0.3, 0.6],
            [0.5, 0.1],
            [0.3, 0.6],
            [0.5, 0.1],
        ]), torch.Tensor([
            [0, 1],
            [0, 0],
            [0, -100],
            [1, 1],
            [0, -100],
            [1, 1],
            [0, -100],
            [1, 1],
        ]), -100, 4),

    ])
    def test_auroc(self, preds, targets, ignore_index, chunks):
        from shared.metrics import CustomAUROC
        from sklearn.metrics import roc_auc_score
        import math

        num_targets = targets.shape[-1] if targets.ndim > 1 else 1
        fn = CustomAUROC(num_targets, ignore_index, return_verbose=True)

        for _preds, _targets in zip(torch.chunk(preds, chunks), torch.chunk(targets, chunks)):
            fn(_preds, _targets)

        auroc_mean, aurocs, _ = fn.compute()

        expected_aurocs = torch.zeros(num_targets)

        if targets.ndim == 1:
            targets = targets.unsqueeze(dim=1)
            preds = preds.unsqueeze(dim=1)

        for t in range(num_targets):

            mask = (targets[:, t] != ignore_index)

            _targets = targets[mask, t]
            _preds = preds[mask, t]

            try:
                expected_auroc = roc_auc_score(_targets, _preds)
            except Exception as e:
                expected_auroc = float("nan")

                print(e)

            expected_aurocs[t] = expected_auroc

            if aurocs[t].isnan():
                assert math.isnan(expected_auroc)
            else:
                assert torch.allclose(torch.tensor(expected_auroc).float(), aurocs[t])

        assert torch.allclose(expected_aurocs[~expected_aurocs.isnan()].mean(), auroc_mean)


class TestThresholdMoving():
    @pytest.mark.parametrize("preds, targets, expected_threshold", [
        (
                torch.Tensor([0.3, 0.2, 0.3, 0.4, 0.0, 0.6, 0.7, 0.8, 0.9, 0.1]),
                torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]), .8
        ),
        (
                torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                torch.Tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), .6
        ),
    ])
    def test_determine(self, preds, targets, expected_threshold):

        from shared.metrics import ThresholdMoving

        th = ThresholdMoving.determine(preds, targets)

        assert torch.allclose(torch.tensor(th), torch.tensor(expected_threshold))




# TODO add tests for sparsity metrics
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
