import logging

from .. import Tensor


class _Function:
    """ Meta class used for creating custom functions that
    support auto backward-propagation.

    Usage:
        Minimal usage requires the user to define the _forward
        and _backward methods in the derived class.
        The implemented function is then called by making calls
        with the created object.
    """

    def __init__(self) -> None:
        self._ctx = list()

    def saveForBackward(self, *args) -> None:
        """ Saves the arguments so they can be used for backward call.

        Args:
            args: Any object that is needed in the backward-pass.
        """
        self._ctx += list(args)

    def getContext(self) -> list:
        """ Get previously saved arguments and clear the list.

        Returns:
            List of all previously saved arguments.
        """
        ctx = list(self._ctx)
        self._ctx = list()
        return ctx

    def _forward(self, *args) -> Tensor:
        """ Performs the operation of the function.

        Usage:
            Needs to be redefiend for each derived class.
            Should not be called directly.

        Args:
            args: One or more Tensors.

        Returns:
            Tensor that represents the result of the operation.
        """
        raise NotImplementedError

    def _backward(self, gradient: Tensor) -> tuple:
        """ Defines the formula for differentiating the
        operation for backward-propagation.

        Usage:
            Needs to be redefiend for each derived class.
            Should not be called directly.

        Args:
            gradient (Tensor): Gradient at the output.

        Returns:
            A tuple of Tensors for each argument passed
            to the _forward method via the args argument.
        """
        raise NotImplementedError

    def backward(self, gradient: Tensor) -> None:
        logging.info(f"Reached function {type(self).__name__} while backward-propagating.")
        logging.debug(f"Input gradient is {gradient}")
        logging.debug(f"Input gradient shape is {gradient.shape}")
        # Calculating gradient for each input
        outGradient = self._backward(gradient)
        logging.debug(f"{type(self).__name__}._backward() returned {outGradient}")
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
