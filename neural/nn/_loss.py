import numpy as np

from .. import Tensor
from ._meta import _Function
from ._activation import LogSoftmax


class F_NLLLoss(_Function):
    """ The negative log likelihood loss.
    Useful to train a classification problem with C classes.

    The input given through a forward call is expected to contain
    log-probabilities of each class. Obtaining log-probabilities
    is easily achieved by adding a LogSoftmax layer in the last
    layer of your network. You may use CrossEntropyLoss instead if
    you prefer not to add an extra layer.

    Args:
        reduction (str): one of 'none', 'mean', 'sum'

    Call Args:
        x (Tensor): (n, C) Tensor
        target (Tensor): (n,) Tensor of integer values in range [0, C-1]

    """

    @staticmethod
    def _forward(ctx, x: Tensor, /, target: Tensor, *, reduction: str = "mean") -> Tensor:
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")

        ctx.saveForBackward(x, target, reduction)
        y = -x[np.arange(x.shape[0]), target]

        if reduction == "mean":
            y = np.mean(y)
        elif reduction == "sum":
            y = np.sum(y)
        elif reduction == "none":
            y = y

        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> Tensor:
        x, target, reduction = ctx.saved
        dx = np.zeros_like(x)
        dx[np.arange(x.shape[0]), target] = -1
        if reduction == "mean":
            dx = dx / x.shape[0]
        dx = dx*gradient
        return Tensor(dx),


class NLLLoss:
    F_NLLLoss.__doc__

    def __init__(self, *, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, x: Tensor, /, target: Tensor) -> Tensor:
        return F_NLLLoss()(x, target=target, reduction=self.reduction)


class F_CrossEntropyLoss:
    """ The cross entropy loss/
    Useful to train a classification problem with C classes.

    The input given through a forward call is expected to contain
    raw, unnormalized scores for each class.

    Args:
        reduction (str): one of 'none', 'mean', 'sum'

    Shapes:
        x (Tensor): (n, C) Tensor
        target (Tensor): (n,) Tensor of integer values in range [0, C-1]

    """

    def __call__(self, x: Tensor, /, *, target: Tensor, reduction: str = "mean") -> Tensor:
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")

        return NLLLoss(reduction=reduction)(LogSoftmax(dim=1)(x), target)


class CrossEntropyLoss:
    F_CrossEntropyLoss.__doc__

    def __init__(self, *, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, x: Tensor, /, target: Tensor) -> Tensor:
        return F_CrossEntropyLoss()(x, target=target, reduction=self.reduction)


class F_L1Loss(_Function):
    """ Creates a criterion that measures the mean absoulute error
    between each element in the input and target.

    Args:
        reduction (str): one of 'none', 'mean', 'sum'

    Shapes:
        x (Tensor): n-dimensional Tensor
        target (Tensor): same shape as the input

    """

    @staticmethod
    def _forward(ctx, x: Tensor, /, target: Tensor, *, reduction: str = "mean") -> Tensor:
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")

        ctx.saveForBackward(x, target, reduction)
        y = np.abs(x - target)

        if reduction == "mean":
            y = np.mean(y)
        elif reduction == "sum":
            y = np.sum(y)
        elif reduction == "none":
            y = y

        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> Tensor:
        x, target, reduction = ctx.saved
        dx = 2*(target < x).astype(float) - 1
        if reduction == "mean":
            dx = dx / x.size
        dx = dx*gradient
        return Tensor(dx),


class L1Loss:
    F_L1Loss.__doc__

    def __init__(self, *, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, x: Tensor, /, target: Tensor) -> Tensor:
        return F_L1Loss()(x, target=target, reduction=self.reduction)
