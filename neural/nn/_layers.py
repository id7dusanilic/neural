import numpy as np

from .. import Tensor
from ._operations import MatMul, Add


class _Layer:
    """ Meta class used for creating neural network layers.

    This class is used for creating neural network layers that
    have a paremeter or a list of parameters that need to be
    optimized during the training of the neural network.

    All operations used within the _Layer need to be derived
    from the _Function class to support auto backward-propagation.

    Usage:
        Minimal usage requires the user to define the _forward
        and the parameters methods in the derived class.
        The implemented function is then called by making calls
        with the created object.
    """

    def parameters(self) -> list:
        """ Get the list of all the layer parameters that
        need to be optimized.

        Usage:
            Needs to be redefiend for each derived class.

        Returns:
            list of Tensors corresponding to all the parameters.
        """
        raise NotImplementedError

    def _forward(self, input_: Tensor) -> Tensor:
        """ Performs the operation of the layer.

        Usage:
            Needs to be redefiend for each derived class.
            Should not be called directly.

        Args:
            input_ (Tensor): input Tensor

        Returns:
            Tensor that represents the result of the layer.
        """
        raise NotImplementedError

    def __call__(self, input_: Tensor) -> Tensor:
        result = self._forward(input_)
        return result

    def __str__(self) -> str:
        result = f"{self.__class__.__name__}("
        for k, v in self.__dict__.items():
            if not isinstance(v, Tensor):
                result += f"{k}={v}, "
        result = f"{result[:-2]})"
        return result


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
        mul = MatMul(rightT=True)(input_, self.weight)
        result = Add()(mul, self.bias) if self._bias else mul
        # Do not cast this to Tensor here.
        # It already is and would overwrite gradFn required for backward-propagation.
        return result
