from typing import List, Optional

import torch
from torchmetrics import Metric
from torchmetrics.functional import auroc, roc


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


class CustomAccuracy(Metric):
    """custom binary accuracy implementation - supports multi target binary cases with defined ignore index"""

    def __init__(self, num_targets: int = 1, ignore_index: int = -100, threshold: float = .5, return_verbose: bool = False,
                 dist_sync_on_step=False):
        """

        Args:
            num_targets (): number of targets
            ignore_index (): index within targets/lables which will be ignored during metric calculation
            threshold ():
            return_verbose (): if true additionally returns individual metric per target
            dist_sync_on_step ():
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=True)

        self.ignore_index = ignore_index
        self.num_targets = num_targets
        self.threshold = threshold
        self.return_verbose = return_verbose

        self.add_state("correct", default=torch.tensor([0] * num_targets), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor([0] * num_targets), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert targets.shape[-1] == self.num_targets if targets.ndim > 1 else True
        assert preds.shape == targets.shape

        if targets.ndim == 1:
            targets = targets.unsqueeze(dim=1)
            preds = preds.unsqueeze(dim=1)

        # assume probabilities
        if preds.ndim == targets.ndim and preds.is_floating_point():
            preds = (preds >= self.threshold).long()

        for t in range(self.num_targets):
            mask = (targets[:, t] != self.ignore_index)

            _preds = preds[mask, t]
            _targets = targets[mask, t]

            self.correct[t] += torch.sum(torch.tensor(_preds == _targets))
            self.total[t] += targets[mask, t].numel()

    def compute(self):
        accuracies = torch.zeros(self.num_targets)

        for i in range(self.num_targets):
            accuracies[i] = self.correct[i].float() / self.total[i] if self.total[i] > 0 else -1

        accuracies = accuracies[(accuracies >= 0)]

        if self.return_verbose:
            return accuracies.mean(), accuracies
        else:
            return accuracies.mean()


class CustomAUROC(Metric):
    """custom AUROC implementation - supports multi target binary cases with definable ignore index"""

    def __init__(self, num_targets: int = 1, ignore_index: int = -100, return_verbose: bool = False, dist_sync_on_step=False):
        """

        Args:
            num_targets (): expected number of targets
            ignore_index (): index within targets/labels which should be ignored
            return_verbose ():
                If true returns individual aurocs per target and the best support threshold or only mean of aurocs (default)

            dist_sync_on_step ():
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=True)

        self.ignore_index = ignore_index
        self.num_targets = num_targets
        self.return_verbose = return_verbose

        self.add_state("preds", default=[], dist_reduce_fx=CustomAUROC.reduce_fx)
        self.add_state("targets", default=[], dist_reduce_fx=CustomAUROC.reduce_fx)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert targets.shape[-1] == self.num_targets if targets.ndim > 1 else True
        assert preds.shape == targets.shape
        assert preds.is_floating_point()

        if targets.ndim == 1:
            targets = targets.unsqueeze(dim=1)
            preds = preds.unsqueeze(dim=1)

        _preds, _targets = [], []
        for t in range(self.num_targets):
            # add only those predictions which we are not supposed to ignore
            mask = (targets[:, t] != self.ignore_index)

            _p = preds[mask, t]
            _t = targets[mask, t]

            if len(self.preds) < t + 1:
                self.preds.append(_p)
                self.targets.append(_t)
            else:
                self.preds[t] = torch.cat([self.preds[t], _p])
                self.targets[t] = torch.cat([self.targets[t], _t])

    def compute(self):
        aurocs = torch.zeros(self.num_targets)
        for t in range(self.num_targets):
            try:
                aurocs[t] = auroc(self.preds[t], self.targets[t].long())
            except Exception as e:
                aurocs[t] = float("nan")

                # print(e)

        if self.return_verbose:
            thresholds = torch.Tensor([ThresholdMoving.determine(self.preds[t], self.targets[t]) for t in range(self.num_targets)])

            return aurocs[~aurocs.isnan()].mean(), aurocs, thresholds
        else:
            return aurocs[~aurocs.isnan()].mean()

    @staticmethod
    def reduce_fx(state: List[torch.Tensor]):
        raise NotImplementedError("reduce fx not yet implemented")


class ThresholdMoving(Metric):
    """
    Custom threshold moving implementation which calculates the best supporting threshold per targets for binary multi target objectives

    Supports multi target binary cases with definable ignore index
    """

    def __init__(self, num_targets: int = 1, ignore_index: int = -100, dist_sync_on_step=False):
        """

        Args:
            num_targets (): expected number of targets
            ignore_index (): index within targets/labels which should be ignored
            dist_sync_on_step ():
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=True)

        self.ignore_index = ignore_index
        self.num_targets = num_targets

        self.add_state("preds", default=[], dist_reduce_fx=CustomAUROC.reduce_fx)
        self.add_state("targets", default=[], dist_reduce_fx=CustomAUROC.reduce_fx)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert targets.shape[-1] == self.num_targets if targets.ndim > 1 else True
        assert preds.shape == targets.shape
        assert preds.is_floating_point()

        if targets.ndim == 1:
            targets = targets.unsqueeze(dim=1)
            preds = preds.unsqueeze(dim=1)

        _preds, _targets = [], []
        for t in range(self.num_targets):
            # add only those predictions which we are not supposed to ignore
            mask = (targets[:, t] != self.ignore_index)

            _p = preds[mask, t]
            _t = targets[mask, t]

            if len(self.preds) < t + 1:
                self.preds.append(_p)
                self.targets.append(_t)
            else:
                self.preds[t] = torch.cat([self.preds[t], _p])
                self.targets[t] = torch.cat([self.targets[t], _t])

    def compute(self):
        thresholds = torch.Tensor([ThresholdMoving.determine(self.preds[t], self.targets[t]) for t in range(self.num_targets)])

        return thresholds

    @staticmethod
    def determine(preds: torch.Tensor, targets: torch.Tensor) -> float:
        """determines the optional threshold for binary classification task"""
        try:
            fpr, tpr, ths = roc(preds, targets)

            j = tpr - fpr

            threshold = ths[torch.argmax(j)].detach().cpu().item()
        except Exception as e:
            threshold = float("nan")

        return threshold

    @staticmethod
    def reduce_fx(state: List[torch.Tensor]):
        raise NotImplementedError("reduce fx not yet implemented")
