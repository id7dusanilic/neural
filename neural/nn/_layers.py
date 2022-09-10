import numpy as np
from scipy.signal import correlate2d

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


class F_Conv2d(_Function):
    """ Applies a 2D convolution over an input signal composed
    of several input planes.

    Args:
        stride (int): stride of the filter
        padding (int): implicit zero padding to be added to both sides

    Shape:
        Input: (N, C, Hin, Win) or (C, Hin, Win) Tensor
        Output: (N, C, Hout, Wout) or (C, Hout, Wout) Tensor

        N is a batch size, C is the number of channels
    """

    @staticmethod
    def _forward(ctx, x: Tensor, w: Tensor, b: Tensor, /, *, padding: int = 0, stride: int = 1) -> Tensor:
        x = x if x.ndim == 4 else x[None, ...]
        ctx.saveForBackward(x, w, b, padding, stride)
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape

        H_ = (H + 2*padding - HH) // stride + 1
        W_ = (W + 2*padding - WW) // stride + 1
        shape_ = N, F, H_, W_
        y = np.zeros(shape_)

        xp = np.pad(x, ((0,), (0,), (padding,), (padding, )), 'constant')
        for n in range(N):
            for f in range(F):
                for c in range(C):
                    y[n, f] += correlate2d(xp[n, c], w[f, c], mode="valid")[..., ::stride, ::stride]

        y = y + b[None, :, None, None] if b is not None else y

        return Tensor(y)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:
        x, w, b, padding, stride = ctx.saved
        N, C, _, _ = x.shape
        F, _, _, _ = w.shape

        def _2dConvXBackwardStride(w, g, stride):
            HG, WG = g.shape
            gStride = np.zeros((stride*HG, stride*WG))
            gStride[::stride, ::stride] = g
            result = correlate2d(gStride, w[::-1, ::-1], mode="full")
            return result

        def _2dConvWeightBackwardStride(x, g, stride):
            HG, WG = g.shape
            HX, WX = x.shape
            gStride = np.zeros((stride*HG, stride*WG))
            gStride[::stride, ::stride] = g
            result = correlate2d(x, gStride, mode="valid")
            result = result[:HX, :WX]
            return result

        xp = np.pad(x, ((0,), (0,), (padding,), (padding, )), "constant")
        dxp = np.zeros_like(xp)
        for n in range(N):
            for f in range(F):
                for c in range(C):
                    dxp[n, c] += _2dConvXBackwardStride(w[f, c], gradient[n, f], stride)
        dx = dxp[..., padding:-padding, padding:-padding]

        dw = np.zeros_like(w)
        for n in range(N):
            for f in range(F):
                for c in range(C):
                    dw[f, c] += _2dConvWeightBackwardStride(xp[n, c], gradient[n, f], stride)

        db = np.sum(gradient, axis=(0, 2, 3)) if b is not None else None

        return dx, dw, db


class Conv2d(_Layer):
    F_Conv2d.__doc__

    def __init__(self, inChannels: int, outChannels: int, kernelSize: tuple[int, int],
            *, bias: bool = True, stride: int = 1, padding: int = 0) -> None:
        super().__init__()

        kernelSize = (kernelSize, kernelSize) if isinstance(kernelSize, int) else kernelSize
        self._bias = bias
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding

        k = np.sqrt(1/(inChannels*kernelSize[0]*kernelSize[1]))
        self.weight = Tensor(2*k*np.random.rand(outChannels, inChannels, kernelSize[0], kernelSize[1]) - k,
                requiresGrad=True)
        self.bias = Tensor(2*k*np.random.rand(outChannels) - k, requiresGrad=True) if bias else None

    def parameters(self) -> list:
        return [self.weight, self.bias] if self._bias else [self.weight]

    def _forward(self, x: Tensor) -> Tensor:
        return F_Conv2d()(x, self.weight, self.bias, padding=self.padding, stride=self.stride)
