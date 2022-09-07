import numpy as np

from .. import Tensor
from ._meta import _Layer
from ._operations import MatMul, Add


class Linear(_Layer):
    """ Applies a linear transformation to the
    incoming data y = x@A.T + b

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

    def __init__(self, inSize: int, outSize: int, bias: bool = True) -> None:
        super().__init__()

        self._bias = bias
        self._inSize = inSize
        self._outSize = outSize

        k_ = np.sqrt(1/inSize)
        self.weight = Tensor(2*k_*np.random.rand(outSize, inSize) - k_, requiresGrad=True)
        self.bias = Tensor(2*k_*np.random.rand(outSize) - k_, requiresGrad=True) if bias else None

    def parameters(self) -> list:
        return [self.weight, self.bias] if self._bias else [self.weight]

    def _forward(self, input_: Tensor) -> Tensor:
        mul = MatMul()(input_, self.weight, rightT=True)
        result = Add()(mul, self.bias) if self._bias else mul
        # Do not cast this to Tensor here.
        # It already is and would overwrite gradFn required for backward-propagation.
        return result
