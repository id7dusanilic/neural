import numpy as np

from ._optim import Optimizer


class Adam(Optimizer):
    """ Implements Adam alghorithm

    Parameters:
        params (list): list of Tensors that will be optimized in each
                       step of the optimizer.
        lr (float): learning rate
        betas (tuple[float, float]) : coefficients used for computing running averages of gradient and its square
        eps (float): term added to the denominator to improve numerical stability
        maximize (bool): If True, maximize instead of minimize
    """

    def __init__(self, params: list, /, *,
            lr: float = 0.001,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-08,
            maximize: bool = False):
        super().__init__(params)

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.maximize = maximize
        self._iterations = 0
        # First momentums
        self._m = [np.zeros_like(param) for param in self.params]
        # Second momentums
        self._v = [np.zeros_like(param) for param in self.params]

    def step(self):
        self._iterations = self._iterations + 1
        for i, (param, m, v) in enumerate(zip(self.params, self._m, self._v)):
            grad = -param.grad if self.maximize else param.grad
            # Updating momentums
            self._m[i] = mnew = self.betas[0]*m + (1 - self.betas[0])*grad
            self._v[i] = vnew = self.betas[1]*v + (1 - self.betas[1])*grad**2
            # Bias correction
            mbc = mnew / (1 - self.betas[0]**self._iterations)
            vbc = vnew / (1 - self.betas[1]**self._iterations)

            paramUpdate = self.lr*mbc / (np.sqrt(vbc) + self.eps)

            param[:] = param[:] - paramUpdate
