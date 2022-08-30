import numpy as np

from .. import Tensor
from ._functions import _Function
from ._activation import LogSoftmax


class NLLLoss(_Function):
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
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")

        self._reduction = reduction

    def _forward(self, input_: Tensor, target: Tensor) -> Tensor:
        self.saveForBackward(input_, target)
        result_ = -input_[np.arange(input_.shape[0]), target]

        if self._reduction == "mean":
            result_ = np.mean(result_)
        elif self._reduction == "sum":
            result_ = np.sum(result_)
        elif self._reduction == "none":
            result_ = result_

        return Tensor(result_)

    def _backward(self, gradient: Tensor) -> Tensor:
        input_, target = self.getContext()
        grad = np.zeros_like(input_)
        grad[np.arange(input_.shape[0]), target] = -1
        if self._reduction == "mean":
            grad = grad / input_.shape[0]
        return Tensor(grad*gradient),


class CrossEntropyLoss:
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

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")

        self._reduction = reduction

    def __call__(self, input_: Tensor, target: Tensor) -> Tensor:
        return NLLLoss(self._reduction)(LogSoftmax(dim=1)(input_), target)

