import numpy as np
from ._tensor import Tensor

class _Function:

    def __init__(self):
        self._ctx = list()

    def saveForBackward(self, *args):
        self._ctx.append(*args)

    def getContext(self):
        return self._ctx
    
    def _forward(self, input_: Tensor) -> Tensor:
        pass
        
    def _backward(self, gradient: Tensor):
        pass

    def backward(self, gradient: Tensor):
        self._backward(gradient)

    def __call__(self, input_: Tensor) -> Tensor:
        result = self._forward(input_)
        result.gradFn = self
        return result


class Linear(_Function):

    def __init__(self, inSize: int, outSize: int, bias: bool = True):
        super().__init__()

        self._bias = bias
        self._inSize = inSize
        self._outSize = outSize

        k_ = np.sqrt(1/inSize)
        self.weight = Tensor(2*k_*np.random.rand(outSize, inSize) - k_, requiresGrad=True)
        self.bias = Tensor(2*k_*np.random.rand(outSize) - k_, requiresGrad=True) if bias else None
    
    def _forward(self, input_: Tensor) -> Tensor:
        self.saveForBackward(input_)
        mul = np.matmul(input_, self.weight.T)
        result = mul + self.bias if self._bias else mul
        return Tensor(result)

    def _backward(self, gradient: Tensor):
        input_, = self.getContext()
        self.weight.grad = Tensor(np.matmul(gradient.T, input_))
        self.bias.grad = Tensor(np.ones_like(self.bias))


def softmax(input_):
    return np.exp(input_) / np.sum(np.exp(input_))


class LogSoftmax(_Function):

    def __init__(self, dim=1):
        super().__init__()

        self.dim = dim

    def _forward(self, input_: Tensor) -> Tensor:
        self.saveForBackward(input_)
        # result = np.log(np.exp(input_) / np.sum(np.exp(input_)))
        result = np.log(softmax(input_))
        return Tensor(result)

    def _backward(self, gradient: Tensor):
        input_, = self.getContext()

