from typing import Union

from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: Union[int, float], num_training_steps, last_epoch=-1):
    """
    adapted from https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup

    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int` or :obj:`float`):
            The number of steps for the warmup phase. if provided as float must be between 0.0 and 1.0 => totalsteps * factor => num_training_steps
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    assert isinstance(num_warmup_steps, int) or (
            num_warmup_steps >= .0 and num_warmup_steps <= 1.0), "num_warmup_steps must be between 0.0 or 1.0 or an integer"
    num_warmup_steps = int(num_training_steps * num_warmup_steps) if isinstance(num_warmup_steps,
                                                                                float) else num_warmup_steps

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_exponential_decay_scheduler(optimizer, decay_rate: float, decay_step: int, last_epoch: int = -1):
    """

    applies a exponential decaying factor to the learning rate - decayed_lr = lr * (decay_rate ^ (current_step / decay_step)

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        decay_rate (:obj:`float`):
            decaying rate
        decay_step (:obj:`int`):
            decaying steps
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Returns:

    """
    def lr_lambda(current_step: int):
        return decay_rate ** (current_step / decay_step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
