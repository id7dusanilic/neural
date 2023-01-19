import numpy as np

from ._optim import Optimizer


class SGD(Optimizer):
    """ Implements stochastic gradient descent (optionally with momentum).

    Parameters:
        params (list): list of Tensors that will be optimized in each
                       step of the optimizer.
        lr (float): learning rate
        momentum (float): momentum factor
        maximize (bool): If True, maximize instead of minimize
    """

    def __init__(self, params: list, /, *,
            lr: float = 0.03,
            momentum: float = 0.0,
            maximize: bool = False):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.maximize = maximize
        self._prevParamsUpdate = [np.zeros_like(param) for param in self.params]

    def step(self):
        for i, (param, prevParamUpdate) in enumerate(zip(self.params, self._prevParamsUpdate)):
            paramUpdate = -self.lr*(self.momentum*prevParamUpdate + param.grad)

            if self.maximize:
                param[:] = param[:] - paramUpdate
            else:
                param[:] = param[:] + paramUpdate

            self._prevParamsUpdate[i] = paramUpdate
