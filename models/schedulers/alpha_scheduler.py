import math
from bisect import bisect_right

__all__ =[
    "StepAlpha", "MultiStepAlpha", "CosAnnealingAlpha", "SegAlpha"
]

class AlphaScheduler(object):
    def __init__(self, lossfunc, last_epoch=-1):
        '''
        control the variation of alpha during training
        NOTE, supposed to be used as follow:
            (othe code)
            for i in range(max_epoch):
                Train_epoch()
                Validate()
                AlphaScheduler.step()
        '''
        if ('peer loss function' not in lossfunc._name) or (not hasattr(lossfunc, "_name")):
            raise ValueError("AlphaScheduler only apply to Peer Loss")
        self._lossfunc = lossfunc
        self._base_alpha = self._lossfunc._alpha
        self._step_count = last_epoch + 1

    def get_alpha(self):
        raise NotImplementedError
    
    def step(self):
        self._step_count += 1
        self._lossfunc._alpha = self.get_alpha()

class StepAlpha(AlphaScheduler):
    '''
    Sets the alpha of Peer Lossf unction to the initial alpha
    decayed by gamma every step_size epochs.
    Args:
        lossfunc: Wrapped loss function.
        step_size (int): Period of alpha decay.
        gamma (float): Multiplicative factor of alpha decay.
            Default: 0.1.
    '''
    def __init__(self, lossfunc, step_size, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepAlpha, self).__init__(lossfunc)

    def get_alpha(self):
        return self._base_alpha * \
               self.gamma ** (self._step_count // self.step_size)

class MultiStepAlpha(AlphaScheduler):
    '''
    Set the alpha of Peer Loss function to the initial alpha decayed
    by gamma once the number of epoch reaches one of the milestones.
    Args:
        lossfunc: Wrapped loss function.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of Alpha decay.
            Default: 0.1.
    '''
    def __init__(self, lossfunc, milestones, gamma=0.1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepAlpha, self).__init__(lossfunc)

    def get_alpha(self):
        return self._base_alpha * \
               self.gamma ** bisect_right(self.milestones, self._step_count)

class CosAnnealingAlpha(AlphaScheduler):
    '''
    Set the alpha of Peer Loss function using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial alpha and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    Args:
        lossfunc: Wrapped loss function.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum alpha value. Default: 0.
    '''
    def __init__(self, lossfunc, T_max, eta_min=0.):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosAnnealingAlpha, self).__init__(lossfunc)
    
    def get_alpha(self):
        return self.eta_min + (self._base_alpha - self.eta_min) * \
               (1 + math.cos(math.pi * self._step_count / self.T_max)) / 2

class SegAlpha(AlphaScheduler):
    '''
    Args:
        lossfunc: Wrapped loss function.
        alpha_list (list): different alphas.
        milestones (list): List of epoch indices. Must be increasing.
    '''
    def __init__(self, lossfunc, alpha_list, milestones, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        if not len(alpha_list) == len(milestones):
            raise ValueError('len(alpha_list) must be equal to len(milestones)')
        super(SegAlpha, self).__init__(lossfunc, last_epoch)
        self.alpha_list = alpha_list
        self.milestones = milestones
        self.len = len(alpha_list)
        self.slope = []
        self.alpha_cur = self._base_alpha
        alpha_cur = self._base_alpha
        epoch_cur = 0
        for i in range(self.len):
            slope = (alpha_list[i] - alpha_cur) / float(milestones[i] - epoch_cur)
            self.slope.append(slope)
            alpha_cur = alpha_list[i]
            epoch_cur = milestones[i]

    def get_alpha(self):
        idx = bisect_right(self.milestones, self._step_count - 1)
        if idx < self.len:
            self.alpha_cur += self.slope[idx]
        return self.alpha_cur