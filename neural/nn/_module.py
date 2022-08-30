from .. import Tensor
from ._functions import _Function
from ._layers import _Layer


class Module:
    """ Meta class used for defining custom neural networks.

    Usage:
        Minimal usage requires the user to define the forward
        method in the derived class.
        The implemented module is then called by making calls
        with the created object.
    """

    def __init__(self) -> None:
        self._parameters = list()
        self._layers = list()

    def __setattr__(self, attrName, attrValue) -> None:
        self.__dict__[attrName] = attrValue

        if isinstance(attrValue, _Layer):
            self._parameters += attrValue.parameters()

        if isinstance(attrValue, (_Layer, _Function)):
            self._layers.append(self.__dict__[attrName])

    def __call__(self, input_: Tensor) -> Tensor:
        return self.forward(input_)

    def __getitem__(self, i: int):
        return self._layers[i]

    def __len__(self) -> int:
        return len(self._layers)

    def parameters(self) -> list:
        return self._parameters

    def zeroGrad(self, setToNone: bool = False) -> None:
        """ Sets the gradients of all parameter Tensors to zero. """
        for param in self._parameters:
            param.zeroGrad(setToNone)

    @staticmethod
    def save(module, filename: str) -> None:
        """ Save the state of a module in a pickle file.

        Args:
            module (Module): module which state is going to be saved
            filename (str): name of the file
        """
        with open(filename, "wb") as f:
            import pickle
            import copy
            module_ = copy.deepcopy(module)
            module_.zeroGrad()
            pickle.dump(module, f)

    @staticmethod
    def load(filename: str):
        """ Load previously saved sate of a module from a
        pickle file.

        Args:
            filename (str): name of the file

        Returns:
            a Module with loaded state
        """
        with open(filename, "rb") as f:
            import pickle
            return pickle.load(f)

    def forward(self, input_: Tensor) -> Tensor:
        """ Computes the output of the module.

        Usage:
            Needs to be redefiend for each derived class.
            Should not be called directly.

        Args:
            input_ (Tensor): input Tensor

        Returns:
            Tensor that represents the output of the module.
        """
        raise NotImplementedError
