import numpy as np

from ._optim import Optimizer


class SGD(Optimizer):
    """ Implements stochastic gradient descent.

    Parameters:
        params (list): list of Tensors that will be optimized in each
                       step of the optimizer.
        lr (float): learning rate
        momentum (float): momentum factor
        weightDecay (float): weight decay (L2 penalty)
        dampening (float): dampening for momentum
        nesterov (bool): enables Nesterov momentum
        maximize (bool): If True, maximize instead of minimize
    """

    def __init__(self, params: list, /, *,
            lr: float = 0.03,
            momentum: float = 0.0,
            dampening: float = 0.0,
            weightDecay: float = 0.0,
            nesterov: bool = False,
            maximize: bool = False):
        super().__init__(params)

        if nesterov and dampening != 0.0:
            raise ValueError("Nesterov requires momentum and zero dampening")

        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weightDecay = weightDecay
        self.nesterov = nesterov
        self.maximize = maximize
        self._prevParamsUpdate = [None] * len(self.params)

    def step(self):
        for param, prevParamUpdate in zip(self.params, self._prevParamsUpdate):
            firstIteration = prevParamUpdate is None
            prevParamUpdate = np.zeros_like(param.grad) if firstIteration else prevParamUpdate

            paramUpdate = param.grad.copy()

            if self.weightDecay != 0.0:
                paramUpdate = paramUpdate + self.weightDecay*param

            if self.momentum != 0.0:
                if not firstIteration:
                    temp = self.momentum*prevParamUpdate + (1-self.dampening)*paramUpdate
                else:
                    temp = paramUpdate

                if self.nesterov:
                    paramUpdate = paramUpdate + self.momentum*temp
                else:
                    paramUpdate = temp

            if self.maximize:
                param[:] = param[:] + self.lr*paramUpdate
            else:
                param[:] = param[:] - self.lr*paramUpdate

            prevParamUpdate = paramUpdate
