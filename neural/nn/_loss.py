import numpy as np

from .. import Tensor
from ._functions import _Function
from ._activation import LogSoftmax


class NLLLoss(_Function):

    def __init__(self, reduction: str = "mean"):
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

    def __init__(self, reduction: str = "mean"):
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")

        self._reduction = reduction


    def __call__(self, input_: Tensor, target: Tensor) -> Tensor:
        return NLLLoss(self._reduction)(LogSoftmax(dim=1)(input_), target)

