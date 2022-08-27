import numpy as np

from .. import Tensor
from ._functions import _Function


class LogSoftmax(_Function):
    """ Applies LofSoftmax function to the input y = log(e**x_i / sum(e**x_k)) """

    def __init__(self, dim):
        """ LogSoftmax function constructor.

        @param dim  axis along which the sum is calculated
        """
        super().__init__()

        self.dim = dim

    def _forward(self, input_: Tensor) -> Tensor:

        def softmax(input_, dim):
            exps = np.exp(input_)
            sum_ = np.sum(exps, axis=self.dim)
            shape_ = list(sum_.shape)
            shape_.insert(dim, 1)
            return exps / sum_.reshape(shape_)

        softmax_ = softmax(input_, self.dim)
        self.saveForBackward(softmax_)
        result = np.log(softmax_)
        return Tensor(result)

    def _backward(self, gradient: Tensor) -> Tensor:
        def formJcb(s):
            return np.eye(s.size) - s

        softmax_, = self.getContext()
        softmax_ = softmax_.swapaxes(-1, self.dim)
        gradientsShape = softmax_.shape
        softmax_ = softmax_.reshape(-1, softmax_.shape[-1])
        gradient_ = gradient.swapaxes(-1, self.dim)
        gradient_ = gradient_.reshape(-1, gradient_.shape[-1])

        gradients = list()
        for s, grad in zip(softmax_, gradient_):
            gradients.append(np.matmul(grad, formJcb(s)))

        return Tensor(gradients).reshape(gradientsShape).swapaxes(-1, self.dim),


class Sigmoid(_Function):

    def _forward(self, input_: Tensor) -> Tensor:
        result_ = 1 / (1 + np.exp(-input_))
        self.saveForBackward(result_)
        return result_

    def _backward(self, gradient: Tensor) -> Tensor:
        sigmoid, = self.getContext()
        grad = sigmoid*(1-sigmoid)
        return Tensor(gradient*grad),


class ReLU(_Function):

    def _forward(self, input_: Tensor) -> Tensor:
        result_ = np.piecewise(input_, [input_ < 0, input_ >= 0], [lambda x: 0, lambda x: x])
        self.saveForBackward(result_)
        return result_

    def _backward(self, gradient: Tensor) -> Tensor:
        result_, = self.getContext()
        grad = (result_ > 0).astype(float)
        return Tensor(gradient*grad),

