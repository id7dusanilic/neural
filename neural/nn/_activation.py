import numpy as np

from .. import Tensor
from ._functions import _Function


class LogSoftmax(_Function):
    """ Applies the log(Softmax(x)) function to an n-dimensional
    input Tensor.

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input

    Args:
        dim (int):  A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0).
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def _forward(self, input_: Tensor) -> Tensor:

        def softmax(input_: Tensor, dim: int):
            """ Help function that applies the Softmax(x) function
            to an n-dimensional input Tensor.

            Shape:
                Input: n-dimensional Tensor
                Output: same shape as Input

            Args:
                dim (int):  A dimension along which LogSoftmax will be computed.

            Returns:
                a Tensor of the same dimension and shape as the input with
                values in the range (0, 1).
            """
            # LogSumExp trick, for numerical stability
            shifted = input_ - np.max(input_)
            exps = np.exp(shifted)
            sum_ = np.sum(exps, axis=self.dim)
            shape_ = list(sum_.shape)
            shape_.insert(dim, 1)
            return exps / sum_.reshape(shape_)

        softmax_ = softmax(input_, self.dim)
        result = np.log(softmax_)
        self.saveForBackward(softmax_)
        return Tensor(result)

    def _backward(self, gradient: Tensor) -> tuple:

        def formJcb(s: Tensor) -> Tensor:
            """ Help function that forms the Jacobian matrix for the
            Log(Softmax(x)) function.

            Shape:
                Input: (n,) Tensor
                Output: (n,n) Tensor

            Args:
                s (Tensor): Result of the Softmax(x) function.

            Returns:
                a Tensor representing the Jacobian matrix.
            """
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
    """ Applies the Sigmoid element-wise function

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input
    """

    def _forward(self, input_: Tensor) -> Tensor:
        result_ = 1 / (1 + np.exp(-input_))
        self.saveForBackward(result_)
        return result_

    def _backward(self, gradient: Tensor) -> tuple:
        sigmoid, = self.getContext()
        grad = sigmoid*(1-sigmoid)
        return Tensor(gradient*grad),


class ReLU(_Function):
    """ Applies the rectified linear unit function element-wise.

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input
    """

    def _forward(self, input_: Tensor) -> Tensor:
        result_ = np.piecewise(input_, [input_ < 0, input_ >= 0], [lambda x: 0, lambda x: x])
        self.saveForBackward(result_)
        return result_

    def _backward(self, gradient: Tensor) -> tuple:
        result_, = self.getContext()
        grad = (result_ > 0).astype(float)
        return Tensor(gradient*grad),

