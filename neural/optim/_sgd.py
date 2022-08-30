from ._optim import Optimizer


class SGD(Optimizer):
    """ Implements stochastic gradient descent.

    Parameters:
        params (list): list of Tensors that will be optimized in each
                       step of the optimizer.
        lr (float): learning rate
        maximize (bool): If True, maximize instead of minimize
    """

    def __init__(self, params: list, lr: float = 0.03, maximize: bool = False):
        super().__init__(params)

        self.lr = lr
        self.maximize = maximize

    def step(self):
        for param in self.params:
            lr = self.lr if self.maximize else -self.lr
            param[:] += lr*param.grad
