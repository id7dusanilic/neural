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

        s = softmax(x, dim)
        ctx.saveForBackward(s, dim)
        y = np.log(s)
        return Tensor(y)

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

        s, dim = ctx.saved
        s = s.swapaxes(-1, dim)
        gradientsShape = s.shape
        s = s.reshape(-1, s.shape[-1])
        gradient_ = gradient.swapaxes(-1, dim)
        gradient_ = gradient_.reshape(-1, gradient_.shape[-1])

        gradients = list()
        for ss, grad in zip(s, gradient_):
            gradients.append(np.matmul(grad, formJcb(ss)))

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
        y = 1 / (1 + np.exp(-x))
        ctx.saveForBackward(y)
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:
        y, = ctx.saved
        dx = gradient*(y*(1-y))
        return Tensor(dx),


class ReLU(_Function):
    """ Applies the rectified linear unit function element-wise.

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input
    """

    @staticmethod
    def _forward(ctx, x: Tensor, /) -> Tensor:
        y = np.piecewise(x, [x < 0, x >= 0], [lambda x: 0, lambda x: x])
        ctx.saveForBackward(y)
        return y

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:
        y, = ctx.saved
        dx = gradient*((y > 0).astype(float))
        return Tensor(dx),


class F_Dropout(_Function):
    """ Applies dropout.

    During training, randomly zeroes some of the elements of the
    input tensor with probability p using samples from a uniform
    distribution. Furthermore, the outputs are scaled by a
    factor of 1 / (1 - p)

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input

    Parameters:
        p (float): probability
    """

    @staticmethod
    def _forward(ctx, x: Tensor, /, *, p: float = 0.5) -> Tensor:
        mask = (np.random.uniform(size=x.shape) > p).astype(np.float) / (1 - p)
        ctx.saveForBackward(mask)
        y = mask * x
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:
        mask, = ctx.saved
        dx = gradient*mask
        return Tensor(dx),

class Dropout:
    F_Dropout.__doc__
    def __init__(self, *, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: Tensor, /) -> Tensor:
        return F_Dropout()(x, p=self.p)
