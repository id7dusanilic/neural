import logging
import numpy as np


class Tensor(np.ndarray):
    """ A numpy array with some additional attributes.

    Tensor.grad         gradient of the vector computed automatically
                        after backward-propagation
    Tensor.gradFn       Function used to compute current vector
    Tensor.requiresGrad If `True` gradient for this Tensor will be
                        calculated during backward-propagation
    """

    def __new__(cls, inputArray, requiresGrad=False):
        obj = np.asarray(inputArray).view(cls)
        obj.grad = None
        obj.gradFn = None
        obj.requiresGrad = requiresGrad
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grad = getattr(obj, "grad", None)
        self.gradFn = getattr(obj, "gradFn", None)
        self.requiresGrad = getattr(obj, "requiresGrad", None)

    def __str__(self):
        result = f"Tensor:\n{super().__str__()}"
        if self.requiresGrad:
            result += ", requiresGrad=True"
        if self.gradFn is not None:
            result += f", gradFn={type(self.gradFn).__name__}"
        return result

    def backward(self, gradient):
        logging.info(f"Reached tensor while backward-propagating")
        logging.debug(self)
        self.grad = gradient
        if self.gradFn is not None and self.requiresGrad:
            self.gradFn.backward(gradient)
