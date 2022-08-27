import numpy as np

from .. import Tensor
from ._functions import *


class _Layer:
    """ Meta class used for creating neural network layers. """

    def parameters(self):
        """ Returns a list of all parameters of the layer that
        need to be optimized.

        Needs to be redefined for each derived class.
        """
        return None

    def _forward(self, input_: Tensor) -> Tensor:
        """ Performs the operation of the layer.

        Needs to be redefiend for each derived class.
        Should not be called directly.
        """
        pass

    def __call__(self, input_: Tensor) -> Tensor:
        result = self._forward(input_)
        return result

    def __str__(self):
        result = f"{self.__class__.__name__}("
        for k, v in self.__dict__.items():
            if not isinstance(v, Tensor):
                result += f"{k}={v}, "
        result = f"{result[:-2]})"
        return result


class Linear(_Layer):
    """ Applies a linear transformation to the incoming data y = x@A.T + b """

    def __init__(self, inSize: int, outSize: int, bias: bool = True):
        """ Linear transformation constructor.

        @param inSize   size of each input sample
        @param outSize  size of each output sample
        @param bias     If set to `False` the layer will not include an additive bias
        """
        super().__init__()

        self._bias = bias
        self._inSize = inSize
        self._outSize = outSize

        k_ = np.sqrt(1/inSize)
        self.weight = Tensor(2*k_*np.random.rand(outSize, inSize) - k_, requiresGrad=True)
        self.bias = Tensor(2*k_*np.random.rand(outSize) - k_, requiresGrad=True) if bias else None

    def parameters(self):
        return [self.weight, self.bias] if self._bias else [self.weight]

    def _forward(self, input_: Tensor) -> Tensor:
        mul = MatMul(rightT=True)(input_, self.weight)
        result = Add()(mul, self.bias) if self._bias else mul
        # Do not cast this to Tensor here.
        # It already is and would overwrite gradFn required for backward-propagation.
        return result
