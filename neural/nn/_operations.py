import numpy as np

from .. import Tensor
from ._functions import _Function


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
