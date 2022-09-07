import numpy as np

from .. import Tensor
from ._meta import _Function


class F_LogSoftmax(_Function):
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

    @staticmethod
    def _forward(ctx, x: Tensor, /, *, dim: int = 1) -> Tensor:

        def softmax(x: Tensor, dim: int):
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
            shifted = x - np.max(x)
            exps = np.exp(shifted)
            sum_ = np.sum(exps, axis=dim)
            shape_ = list(sum_.shape)
            shape_.insert(dim, 1)
            return exps / sum_.reshape(shape_)

        softmax_ = softmax(x, dim)
        result = np.log(softmax_)
        ctx.saveForBackward(softmax_, dim)
        return Tensor(result)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:

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

        softmax_, dim = ctx.saved
        softmax_ = softmax_.swapaxes(-1, dim)
        gradientsShape = softmax_.shape
        softmax_ = softmax_.reshape(-1, softmax_.shape[-1])
        gradient_ = gradient.swapaxes(-1, dim)
        gradient_ = gradient_.reshape(-1, gradient_.shape[-1])

        gradients = list()
        for s, grad in zip(softmax_, gradient_):
            gradients.append(np.matmul(grad, formJcb(s)))

        return Tensor(gradients).reshape(gradientsShape).swapaxes(-1, dim),


class LogSoftmax:
    F_LogSoftmax.__doc__

    def __init__(self, *, dim: int = 1) -> None:
        self.dim = dim

    def __call__(self, x: Tensor, /) -> Tensor:
        return F_LogSoftmax()(x, dim=self.dim)


class Sigmoid(_Function):
    """ Applies the Sigmoid element-wise function

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input
    """

    @staticmethod
    def _forward(ctx, x: Tensor, /) -> Tensor:
        result_ = 1 / (1 + np.exp(-x))
        ctx.saveForBackward(result_)
        return result_

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:
        sigmoid, = ctx.saved
        grad = sigmoid*(1-sigmoid)
        return Tensor(gradient*grad),


class ReLU(_Function):
    """ Applies the rectified linear unit function element-wise.

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input
    """

    @staticmethod
    def _forward(ctx, x: Tensor, /) -> Tensor:
        result_ = np.piecewise(x, [x < 0, x >= 0], [lambda x: 0, lambda x: x])
        ctx.saveForBackward(result_)
        return result_

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:
        result_, = ctx.saved
        grad = (result_ > 0).astype(float)
        return Tensor(gradient*grad),
