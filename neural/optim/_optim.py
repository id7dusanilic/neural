class Optimizer:
    """ Meta class used for creating custom optimizers.

    Usage:
        Minimal usage requires the user to define the step
        method in the derived class.
    """

    def __init__(self, params: list) -> None:
        if not isinstance(params, list):
            raise TypeError(f"params arguments should be a list, got {type(params)} instead")
        if len(params) == 0:
            raise ValueError("Optimizer got an empty paramter list")

        self.params = params

    def zeroGrad(self, setToNone: bool = False) -> None:
        """ Sets the gradients of all optimized Tensors to zero. """
        for param in self.params:
            param.zeroGrad(setToNone)

    def step(self, closure: callable):
        """ Perfroms a single optimization step

        Args:
            closure (callable): A closure that reevaluates the model
            and returns the loss. Optional.
        """
        raise NotImplementedError
