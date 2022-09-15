import numpy as np

from ._optim import Optimizer


class SGD(Optimizer):
    """ Implements stochastic gradient descent.

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
        self._prevParamsUpdate = [None] * len(self.params)

    def step(self):
        for i, (param, prevParamUpdate) in enumerate(zip(self.params, self._prevParamsUpdate)):
            firstIteration = prevParamUpdate is None
            prevParamUpdate = np.zeros_like(param.grad) if firstIteration else prevParamUpdate

            if self.momentum != 0.0:
                if not firstIteration:
                    paramUpdate = self.momentum*prevParamUpdate - self.lr*param.grad
                else:
                    paramUpdate = -self.lr*param.grad
            else:
                paramUpdate = -self.lr*param.grad

            if self.maximize:
                param[:] = param[:] - paramUpdate
            else:
                param[:] = param[:] + paramUpdate

            self._prevParamsUpdate[i] = paramUpdate
