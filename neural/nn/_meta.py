import logging
from copy import deepcopy

from .. import Tensor


class _Context:

    def __init__(self) -> None:
        self._saved = list()

    def saveForBackward(self, *args) -> None:
        """ Saves the arguments so they can be used for backward call.

        Args:
            args: Any object that is needed in the backward-pass.
        """
        self._saved += deepcopy(list(args))

    @property
    def saved(self) -> list:
        """ Get previously saved arguments and clear the list.

        Returns:
            List of all previously saved arguments.
        """
        saved = list(self._saved)
        self._saved = list()
        return saved


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
        self._ctx = _Context()

    @staticmethod
    def _forward(ctx, *args) -> Tensor:
        """ Performs the operation of the function.

        Usage:
            Needs to be redefiend for each derived class.
            Should not be called directly.

        Args:
            ctx (_Context): context to save to for backward pass
            args: One or more Tensors.

        Returns:
            Tensor that represents the result of the operation.
        """
        raise NotImplementedError

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple:
        """ Defines the formula for differentiating the
        operation for backward-propagation.

        Usage:
            Needs to be redefiend for each derived class.
            Should not be called directly.

        Args:
            ctx (_Context): context saved in the forward pass
            gradient (Tensor): Gradient at the output.

        Returns:
            A tuple of Tensors for each argument passed
            to the _forward method via the args argument.
        """
        raise NotImplementedError

    def backward(self, gradient: Tensor) -> None:
        cls = self.__class__
        logging.info(f"Reached function {type(self).__name__} while backward-propagating.")
        logging.debug(f"Input gradient is {gradient}")
        logging.debug(f"Input gradient shape is {gradient.shape}")
        # Calculating gradient for each input
        outGradient = cls._backward(self._ctx, gradient.reshape(self._outShape))
        logging.debug(f"{type(self).__name__}._backward() returned {outGradient}")
        # Continuing backward-propagation down the graph
        for tensor, grad in zip(self._tensors, outGradient):
            if tensor is not None:
                if tensor.requiresGrad:
                    tensor.backward(grad)

    def __call__(self, *tensors, **kwargs) -> Tensor:
        cls = self.__class__
        # Saving input Tensors for automatic backward-propagation
        self._tensors = tensors
        # Performing the calculation
        result = cls._forward(self._ctx, *tensors, **kwargs)
        # Setting the gradient function for automatic backward-propagation
        result.gradFn = self
        # The output requires gradient calculation implicitly
        result.requiresGrad = True
        # Saving output shape
        self._outShape = result.shape
        return result


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

    def _forward(self, x: Tensor) -> Tensor:
        """ Performs the operation of the layer.

        Usage:
            Needs to be redefiend for each derived class.
            Should not be called directly.

        Args:
            x (Tensor): input Tensor

        Returns:
            Tensor that represents the result of the layer.
        """
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        result = self._forward(x)
        return result

    def __str__(self) -> str:
        result = f"{self.__class__.__name__}("
        for k, v in self.__dict__.items():
            if not isinstance(v, Tensor):
                result += f"{k}={v}, "
        result = f"{result[:-2]})"
        return result
