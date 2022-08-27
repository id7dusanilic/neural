import logging
import numpy as np

from .. import Tensor


class _Function:
    """ Meta class used for creating custom functions that support auto backward-propagation. """

    def __init__(self):
        self._ctx = list()

    def saveForBackward(self, *args):
        """ Saves the arguments so they can be used for backward call. """
        self._ctx += list(args)

    def getContext(self) -> list:
        """ Returns previously saved arguments with the saveForBackward function. """
        return self._ctx
    
    def _forward(self, *args) -> Tensor:
        """ Performs the operation of the function.

        Needs to be redefiend for each derived class.
        Should not be called directly.

        Should take a number of inputs, and return an output tensor.
        """
        pass
        
    def _backward(self, gradient: Tensor) -> Tensor:
        """ Defines the formula for differentiating the operation for
        backward-propagation.

        Needs to be redefined for each derived class.
        Should not be called directly.

        Should return a tensor for each input that is passed to the
        _forward method.
        """
        pass

    def backward(self, gradient: Tensor):
        logging.info(f"Reached function {type(self).__name__} while backward-propagating.")
        logging.info(f"Input gradient is {gradient}")
        logging.info(f"Input gradient shape is {gradient.shape}")
        # Calculating gradient for each input
        outGradient = self._backward(gradient)
        logging.info(f"{type(self).__name__}._backward() returned {outGradient}")
        # Continuing backward-propagation down the graph
        for arg, grad in zip(self._args, outGradient):
            if arg.requiresGrad:
                arg.backward(grad)

    def __call__(self, *args) -> Tensor:
        # Saving input Tensors for automatic backward-propagation
        self._args = args
        # Performing the calculation
        result = self._forward(*args)
        # Setting the gradient function for automatic backward-propagation
        result.gradFn = self
        # The output requires gradient calculation implicitly
        result.requiresGrad = True
        return result


class Add(_Function):
    """ Applies addition of two Tensors. """

    def _forward(self, left: Tensor, right: Tensor) -> Tensor:
        self.saveForBackward(left, right)
        add = left + right
        return Tensor(add)

    def _backward(self, gradient: Tensor) -> Tensor:
        left, right = self.getContext()
        jcbLeft, jcbRight = np.ones_like(left), np.ones_like(right)
        return Tensor(gradient*jcbLeft), Tensor(gradient*jcbRight)


class MatMul(_Function):
    """ Applies matrix multiplication of two Tensors, with option to prior transpose either. """

    def __init__(self, leftT: bool = False, rightT: bool = False):
        """ MatMul function constructor.

        @param leftT    if `True` left matrix will be transposed prior to multiplication
        @param rightT   if `True` right matrix will be transposed prior to multiplication
        """
        super().__init__()

        self.leftT = leftT
        self.rightT = rightT

    def _forward(self, left: Tensor, right: Tensor) -> Tensor:
        left_ = left.T if self.leftT else left
        right_ = right.T if self.rightT else right
        self.saveForBackward(left_.T, right_.T)
        mul = np.matmul(left_, right_)
        return Tensor(mul)

    def _backward(self, gradient: Tensor) -> Tensor:
        left, right = self.getContext()
        gradientLeft = np.matmul(gradient, right)
        gradientLeft = gradientLeft.T if self.leftT else gradientLeft
        gradientRight = np.matmul(left, gradient)
        gradientRight = gradientRight.T if self.rightT else gradientRight
        return gradientLeft, gradientRight


class LogSoftmax(_Function):
    """ Applies LofSoftmax function to the input y = log(e**x_i / sum(e**x_k)) """

    def __init__(self, dim):
        """ LogSoftmax function constructor.
        
        @param dim  axis along which the sum is calculated
        """
        super().__init__()

        self.dim = dim

    def _forward(self, input_: Tensor) -> Tensor:

        def softmax(input_, dim):
            exps = np.exp(input_)
            sum_ = np.sum(exps, axis=self.dim)
            shape_ = list(sum_.shape)
            shape_.insert(dim, 1)
            return exps / sum_.reshape(shape_)

        softmax_ = softmax(input_, self.dim)
        self.saveForBackward(softmax_)
        result = np.log(softmax_)
        return Tensor(result)

    def _backward(self, gradient: Tensor) -> Tensor:
        def formJcb(s):
            return np.eye(s.size) - s

        softmax_, = self.getContext()
        softmax_ = softmax_.swapaxes(-1, self.dim)
        resultsShape = softmax_.shape
        softmax_ = softmax_.reshape(-1, softmax_.shape[-1])
        gradient_ = gradient.swapaxes(-1, self.dim)
        gradient_ = gradient_.reshape(-1, gradient_.shape[-1])

        results = list()
        for s, grad in zip(softmax_, gradient_):
            results.append(np.matmul(grad, formJcb(s)))

        return Tensor(results).reshape(resultsShape).swapaxes(-1, self.dim),
