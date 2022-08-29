from .. import Tensor


class Optimizer:

    def __init__(self, params: list):

        if not isinstance(params, list):
            raise TypeError(f"params arguments should be a list, got {type(params)} instead")

        if len(params) == 0:
            raise ValueError("Optimizer got an empty paramter list")

        self.params = params


    def zeroGrad(self, setToNone: bool = False):
        """ Sets the gradients of all optimized `Tensor`s to zero. """
        for param in self.params:
            param.zeroGrad(setToNone)

    def step(self, closure: callable):
        """ Perfroms a single optimization step

        Args:
            closure (callable): A closure that reevaluates the model
            and returns the loss. Optional.
        """
        raise NotImplementedError
