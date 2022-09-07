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
        input_ (Tensor): (n, C) Tensor
        target (Tensor): (n,) Tensor of integer values in range [0, C-1]

    """

    @staticmethod
    def _forward(ctx, input_: Tensor, /, target: Tensor, *, reduction: str = "mean") -> Tensor:
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")

        ctx.saveForBackward(input_, target, reduction)
        result_ = -input_[np.arange(input_.shape[0]), target]

        if reduction == "mean":
            result_ = np.mean(result_)
        elif reduction == "sum":
            result_ = np.sum(result_)
        elif reduction == "none":
            result_ = result_

        return Tensor(result_)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> Tensor:
        input_, target, reduction = ctx.saved
        grad = np.zeros_like(input_)
        grad[np.arange(input_.shape[0]), target] = -1
        if reduction == "mean":
            grad = grad / input_.shape[0]
        return Tensor(grad*gradient),


class NLLLoss:
    F_NLLLoss.__doc__

    def __init__(self, *, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, input_: Tensor, /, target: Tensor) -> Tensor:
        return F_NLLLoss()(input_, target=target, reduction=self.reduction)


class F_CrossEntropyLoss:
    """ The cross entropy loss/
    Useful to train a classification problem with C classes.

    The input given through a forward call is expected to contain
    raw, unnormalized scores for each class.

    Args:
        reduction (str): one of 'none', 'mean', 'sum'

    Call Args:
        input_ (Tensor): (n, C) Tensor
        target (Tensor): (n,) Tensor of integer values in range [0, C-1]

    """

    def __call__(self, input_: Tensor, /, *, target: Tensor, reduction: str = "mean") -> Tensor:
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")

        return NLLLoss(reduction=reduction)(LogSoftmax(dim=1)(input_), target)


class CrossEntropyLoss:
    F_CrossEntropyLoss.__doc__

    def __init__(self, *, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, input_: Tensor, /, target: Tensor) -> Tensor:
        return F_CrossEntropyLoss()(input_, target=target, reduction=self.reduction)
