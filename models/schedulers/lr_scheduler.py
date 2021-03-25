import math
import torch.optim.lr_scheduler as lr_scheduler
from bisect import bisect_right
from collections import Counter
import warnings

__all__ = [
    "StepLR", "MultiStepLR", "CosineAnnealingLR",
    # "LrCosineScheduler",
]


class StepLR(lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.alpha = 0.0
        self.record_lr = [group['lr'] for group in optimizer.param_groups]
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # if not self._get_lr_called_within_step:
        #     warnings.warn("To get the last learning rate computed by the scheduler, "
        #                   "please use `get_last_lr()`.", DeprecationWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            # return [group['lr'] for group in self.optimizer.param_groups]
            return [cur_lr / (1.0+self.alpha) for cur_lr in self.record_lr]
        
        self.record_lr = [cur_lr * self.gamma for cur_lr in self.record_lr]
        # return [group['lr'] * self.gamma
        #         for group in self.optimizer.param_groups]
        return [cur_lr / (1.0+self.alpha) for cur_lr in self.record_lr]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

class MultiStepLR(lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.alpha = 1.0
        self.record_lr = [group['lr'] for group in optimizer.param_groups]
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # if not self._get_lr_called_within_step:
        #     warnings.warn("To get the last learning rate computed by the scheduler, "
        #                   "please use `get_last_lr()`.", DeprecationWarning)

        if self.last_epoch not in self.milestones:
            # return [group['lr'] for group in self.optimizer.param_groups]
            return [cur_lr / (1.0+self.alpha) for cur_lr in self.record_lr]

        self.record_lr = [cur_lr * self.gamma ** self.milestones[self.last_epoch] 
                            for cur_lr in self.record_lr]
        # return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
        #         for group in self.optimizer.param_groups]
        return [cur_lr / (1.0+self.alpha) for cur_lr in self.record_lr]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]

class CosineAnnealingLR(lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        T_{cur} \neq (2k+1)T_{max};\\
        \eta_{t+1} = \eta_{t} + (\eta_{max} - \eta_{min})\frac{1 -
        \cos(\frac{1}{T_{max}}\pi)}{2},
        T_{cur} = (2k+1)T_{max}.\\

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.alpha = 0.0
        self.record_lr = [group['lr'] for group in optimizer.param_groups]
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def restart(self):
        r'''NOTE call before step()'''
        self.last_epoch = -1

    def get_lr(self):
        # if not self._get_lr_called_within_step:
        #     warnings.warn("To get the last learning rate computed by the scheduler, "
        #                   "please use `get_last_lr()`.", DeprecationWarning)

        if self.last_epoch == 0:
            return [base_lr / (1.0 + self.alpha) for base_lr in self.base_lrs]

        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            self.record_lr = [group['lr'] + (base_lr - self.eta_min) *
                                (1 - math.cos(math.pi / self.T_max)) / 2
                                for base_lr, group in
                                zip(self.base_lrs, self.optimizer.param_groups)]
            return [cur_lr / (1.0+self.alpha) for cur_lr in self.record_lr]

        self.record_lr = [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                            (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                            (group['lr'] - self.eta_min) + self.eta_min
                            for group in self.optimizer.param_groups]
        return [cur_lr / (1.0+self.alpha) for cur_lr in self.record_lr]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


#NOTE deprecated
# class LrCosineScheduler(lr_scheduler._LRScheduler):
#     r"""
#     Cosine scheduler for learning rate;
#     adopted from ECCV18 deep co-training paper
#     """
    
#     def __init__(self, optimizer, T_max, last_epoch=-1):
#         self.T_max = T_max
#         self.alpha = 0.0
#         super(LrCosineScheduler, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         return [ base_lr * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (1.0+self.alpha)
#                 for base_lr in self.base_lrs]