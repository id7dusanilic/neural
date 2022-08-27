import logging

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
