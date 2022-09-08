import numpy as np

from .. import Tensor
from ._meta import _Layer, _Function
from ._operations import MatMul, Add


class F_Linear(_Function):
    """ Applies a linear transformation to the
    incoming data y = x@w.T + b

    Args:
        inSize: size of each input sample
        outSize: size of each output sample
        bias: if set to `False`, the layer will not include an additive bias.

    Shape:
        Input: (n, inSize) Tensor
        Output: (n, OutSize) Tensor

    Parameters:
        weight (Tensor): weights of the layer of shape (outSize, inSize).
        bias (Tensor): bias of the layer of shape (outSize,).
    """

    @staticmethod
    def _forward(ctx, x: Tensor, w: Tensor, b: Tensor, /) -> Tensor:
        ctx.saveForBackward(x, w, b)
        y = np.matmul(x, w.T) + b if b is not None else np.matmul(x, w.T)
        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:
        x, w, b = ctx.saved
        dx = np.dot(gradient, w)
        dw = np.dot(x.T, gradient).T
        db = np.sum(gradient, axis=0) if b is not None else None
        return dx, dw, db


class Linear(_Layer):
    F_Linear.__doc__

    def __init__(self, inSize: int, outSize: int, *, bias: bool = True) -> None:
        super().__init__()

        self._bias = bias
        self.inSize = inSize
        self.outSize = outSize

        k = np.sqrt(1/inSize)
        self.weight = Tensor(2*k*np.random.rand(outSize, inSize) - k, requiresGrad=True)
        self.bias = Tensor(2*k*np.random.rand(outSize) - k, requiresGrad=True) if bias else None

    def parameters(self) -> list:
        return [self.weight, self.bias] if self._bias else [self.weight]

    def _forward(self, x: Tensor) -> Tensor:
        return F_Linear()(x, self.weight, self.bias)
        super().__init__()

        self._bias = bias
        self._inSize = inSize
        self._outSize = outSize

        k_ = np.sqrt(1/inSize)
        self.weight = Tensor(2*k_*np.random.rand(outSize, inSize) - k_, requiresGrad=True)
        self.bias = Tensor(2*k_*np.random.rand(outSize) - k_, requiresGrad=True) if bias else None

    def parameters(self) -> list:
        return [self.weight, self.bias] if self._bias else [self.weight]

    def _forward(self, x: Tensor) -> Tensor:
        return F_Linear()(x, self.weight, self.bias)
