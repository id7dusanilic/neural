import numpy as np
from scipy.ndimage import maximum_filter

from .. import Tensor
from ._meta import _Function


def softmax(x: Tensor, dim: int = 1) -> Tensor:
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
        s = softmax(x, dim)
        ctx.saveForBackward(s, dim)
        y = np.log(s)
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple[Tensor, ...]:

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
            return Tensor(np.eye(s.size) - s)

        s, dim = ctx.saved
        s = s.swapaxes(-1, dim)
        gradientsShape = s.shape
        s = s.reshape(-1, s.shape[-1])
        gradient_ = gradient.swapaxes(-1, dim)
        gradient_ = gradient_.reshape(-1, gradient_.shape[-1])

        gradients = list()
        for ss, grad in zip(s, gradient_):
            gradients.append(np.matmul(grad, formJcb(ss)))

        gradients = np.array(gradients).reshape(gradientsShape).swapaxes(-1, dim)

        return Tensor(gradients),


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
        y = np.empty_like(x)
        y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
        y[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
        ctx.saveForBackward(y)
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple[Tensor, ...]:
        y, = ctx.saved
        dx = gradient * (y*(1 - y))  # noqa: E226
        return Tensor(dx),


class Tanh(_Function):
    """ Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input
    """

    @staticmethod
    def _forward(ctx, x: Tensor, /) -> Tensor:
        expx = np.exp(x)
        expxn = np.exp(-x)
        y = (expx - expxn) / (expx + expxn)
        ctx.saveForBackward(y)
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple[Tensor, ...]:
        y, = ctx.saved
        dx = gradient * (1 - y**2)
        return Tensor(dx),


class ReLU(_Function):
    """ Applies the rectified linear unit function element-wise.

    Shape:
        Input: n-dimensional Tensor
        Output: same shape as Input
    """

    @staticmethod
    def _forward(ctx, x: Tensor, /) -> Tensor:
        y = np.piecewise(x, [x < 0, x >= 0], [lambda _: 0, lambda x: x])
        ctx.saveForBackward(y)
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple[Tensor, ...]:
        y, = ctx.saved
        dx = gradient * ((y > 0).astype(float))
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
        mask = (np.random.uniform(size=x.shape) > p).astype(np.float64) / (1 - p)
        ctx.saveForBackward(mask)
        y = mask * x
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple[Tensor, ...]:
        mask, = ctx.saved
        dx = gradient * mask
        return Tensor(dx),


class Dropout:
    F_Dropout.__doc__

    def __init__(self, *, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: Tensor, /) -> Tensor:
        return F_Dropout()(x, p=self.p)


class F_MaxPool2d(_Function):
    """ Applies a 2D max pooling over an input signal composed
    of several input planes.

    Args:
        kernelSize (int, tuple[int, int]): the size if the window
        stride (int): stride of the window
        padding (int): implicit zero padding to be added to both sides

    Shape:
        Input: (N, C, Hin, Win) or (C, Hin, Win) Tensor
        Output: (N, C, Hout, Wout) or (C, Hout, Wout) Tensor

        N is a batch size, C is the number of channels
    """

    @staticmethod
    def _forward(ctx, x: Tensor, kernelSize: tuple[int, int], /, *, padding: int = 0, stride: int = 1) -> Tensor:
        x = x if x.ndim == 4 else x[None, ...]
        kernelSize = (kernelSize, kernelSize) if isinstance(kernelSize, int) else kernelSize
        N, C, _, _ = x.shape

        xp = np.pad(x, ((0,), (0,), (padding,), (padding, )), "constant")
        yp = np.zeros_like(xp)

        for n in range(N):
            for c in range(C):
                yp[n, c] = maximum_filter(xp[n, c], size=kernelSize, mode="constant")

        xstart, ystart = kernelSize[0] // 2, kernelSize[1] // 2
        xstop = -xstart if kernelSize[0] % 2 != 0 else -xstart + 1
        xstop = None if xstop == 0 else xstop
        ystop = -ystart if kernelSize[1] % 2 != 0 else -ystart + 1
        ystop = None if ystop == 0 else ystop

        y = yp[..., xstart:xstop:stride, ystart:ystop:stride]

        ctx.saveForBackward(x, y)
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple[Tensor, ...]:
        x, y = ctx.saved
        dx = np.zeros_like(x)
        for uniq in np.unique(y):
            dx[x == uniq] = np.count_nonzero(y == uniq) * np.mean(gradient[y == uniq])
        return Tensor(dx),


class MaxPool2d:
    F_MaxPool2d.__doc__

    def __init__(self, kernelSize: tuple[int, int], /, *, padding: int = 0, stride: int = 1) -> None:
        kernelSize = (kernelSize, kernelSize) if isinstance(kernelSize, int) else kernelSize
        self.kernelSize = kernelSize
        self.padding = padding
        self.stride = stride

    def __call__(self, x: Tensor, /) -> Tensor:
        return F_MaxPool2d()(x, self.kernelSize, padding=self.padding, stride=self.stride)
