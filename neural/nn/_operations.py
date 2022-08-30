import numpy as np

from .. import Tensor
from ._functions import _Function


class Add(_Function):
    """ Applies addition of two Tensors out = left + right.

    Shape:
        Input:  both operands should have the same dimension, or
                they have to be broadcastable to a common dimension.
        Output: same as the input operands, or the common
                broadcastable dimension.
    """

    def _forward(self, left: Tensor, right: Tensor) -> Tensor:
        self.saveForBackward(left, right)
        add = left + right
        return Tensor(add)

    def _backward(self, gradient: Tensor) -> tuple:
        left, right = self.getContext()
        gradientLeft = Tensor(gradient*np.ones_like(left))
        gradientRight = Tensor(gradient*np.ones_like(right))

        def collapseIfNeccesary(x, grad):
            # Broadcasting happened because left.shape != right.shape
            if x.shape != grad.shape:
                shape_ = np.ones_like(x.shape if x.ndim > grad.ndim else grad.shape)
                xShape = shape_.copy(); xShape[-len(x.shape):] = x.shape
                gradShape = shape_.copy(); gradShape[-len(grad.shape):] = grad.shape
                axis_ = np.argwhere(np.array(xShape) != np.array(gradShape))
                axis_ = axis_.item() if axis_.size == 1 else None
                grad = np.sum(grad, axis=axis_) if axis_ is not None else grad

            return grad

        gradientLeft = collapseIfNeccesary(left, gradientLeft)
        gradientRight = collapseIfNeccesary(right, gradientRight)

        return gradientLeft, gradientRight


class MatMul(_Function):
    """ Applies matrix multiplication of two Tensors,
    with option to prior transpose either. out = left(.T)@right(.T)

    Shapes:
        Input: dimensions suitable for matrix multiplicaiton.

    Args:
        leftT (bool): if True left matrix will be transposed prior to multiplication
        rightT (bool): if True right matrix will be transposed prior to multiplication
    """

    def __init__(self, leftT: bool = False, rightT: bool = False) -> None:
        super().__init__()

        self.leftT = leftT
        self.rightT = rightT

    def _forward(self, left: Tensor, right: Tensor) -> Tensor:
        left_ = left.T if self.leftT else left
        right_ = right.T if self.rightT else right
        self.saveForBackward(left_.T, right_.T)
        mul = np.matmul(left_, right_)
        return Tensor(mul)

    def _backward(self, gradient: Tensor) -> tuple:
        left, right = self.getContext()
        gradientLeft = np.matmul(gradient, right)
        gradientLeft = gradientLeft.T if self.leftT else gradientLeft
        gradientRight = np.matmul(left, gradient)
        gradientRight = gradientRight.T if self.rightT else gradientRight
        return gradientLeft, gradientRight
