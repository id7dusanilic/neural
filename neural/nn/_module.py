from .. import Tensor
from ._functions import _Function
from ._layers import _Layer


class Module:
    """ Meta class used for defining custom neural networks. """

    def __init__(self):
        self.parameters = list()
        self.layers = list()

    def __setattr__(self, attrName, attrValue):
        self.__dict__[attrName] = attrValue

        if isinstance(attrValue, _Layer):
            self.parameters += attrValue.parameters()

        if isinstance(attrValue, (_Layer, _Function)):
            self.layers.append(self.__dict__[attrName])

    def __call__(self, input_: Tensor) -> Tensor:
        return self.forward(input_)

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def zeroGrad(self, setToNone: bool = False):
        """ Sets the gradients of all parameter `Tensor`s to zero. """
        for param in self.parameters:
            param.zeroGrad(setToNone)

    @staticmethod
    def save(module, filename: str) -> None:
        with open(filename, "wb") as f:
            import pickle
            import copy
            module_ = copy.deepcopy(module)
            module_.zeroGrad()
            pickle.dump(module, f)

    @staticmethod
    def load(filename: str) -> None:
        with open(filename, "rb") as f:
            import pickle
            return pickle.load(f)

    def forward(self, input_: Tensor) -> Tensor:
        """ Calculates the output of the module.

        Needs to be redefined for each derived class.
        Should not be called directly.
        """
        raise NotImplementedError
