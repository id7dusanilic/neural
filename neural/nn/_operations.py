import numpy as np

from .. import Tensor
from ._meta import _Function


class Add(_Function):
    """ Applies addition of two Tensors out = left + right.

    Shape:
        Input:  both operands should have the same dimension, or
                they have to be broadcastable to a common dimension.
        Output: same as the input operands, or the common
                broadcastable dimension.
    """

    @staticmethod
    def _forward(ctx, left: Tensor, right: Tensor, /) -> Tensor:
        ctx.saveForBackward(left, right)
        add = left + right
        return Tensor(add)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple[Tensor, ...]:
        left, right = ctx.saved
        gradientLeft = Tensor(gradient * np.ones_like(left))
        gradientRight = Tensor(gradient * np.ones_like(right))

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

    @staticmethod
    def _forward(ctx, left: Tensor, right: Tensor, /, *, leftT: bool = False, rightT: bool = False) -> Tensor:
        left_ = left.T if leftT else left
        right_ = right.T if rightT else right
        ctx.saveForBackward(left_, right_, leftT, rightT)
        mul = np.matmul(left_, right_)
        return Tensor(mul)

    @staticmethod
    def _backward(ctx, gradient: Tensor) -> tuple[Tensor, ...]:
        left, right, leftT, rightT = ctx.saved
        gradientLeft = np.dot(gradient, right.T)
        gradientLeft = gradientLeft.T if leftT else gradientLeft
        gradientRight = np.dot(left.T, gradient)
        gradientRight = gradientRight.T if rightT else gradientRight
        return gradientLeft, gradientRight
