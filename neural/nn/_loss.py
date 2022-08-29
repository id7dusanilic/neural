import numpy as np

from .. import Tensor
from ._functions import _Function
from ._activation import LogSoftmax


class NLLLoss(_Function):

    def _forward(self, input_: Tensor, target: Tensor) -> Tensor:
        self.saveForBackward(input_, target)
        result_ = -input_[np.arange(input_.shape[0]), target]
        return Tensor(result_)

    def _backward(self, gradient: Tensor) -> Tensor:
        input_, target = self.getContext()
        grad = np.zeros_like(input_)
        grad[np.arange(input_.shape[0]), target] = -1
        return Tensor(grad*gradient),


class CrossEntropyLoss:

    def __call__(self, input_: Tensor, target: Tensor) -> Tensor:
        return NLLLoss()(LogSoftmax(dim=1)(input_), target)

