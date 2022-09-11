import numpy as np

from . import Tensor


class IDXFormatLoader:

    DECODE = [
        None, None, None, None, None, None, None, None,
        np.uint8,
        np.int8,
        None,
        np.int16,
        np.int32,
        np.float32,
        np.float64    
    ]

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            _, _, code, ndim = f.read(4)
            dtype = cls.DECODE[code]
            dims = [int.from_bytes(f.read(4), byteorder="big") for _ in range(ndim)]
            
            data = np.frombuffer(f.read(np.prod(dims)), dtype=dtype)
            data = data.reshape(dims)
            
            return Tensor(data)


class MNIST:

    URL = "http://yann.lecun.com/exdb/mnist/"
    TRAIN_IMAGES = "train-images-idx3-ubyte"
    TRAIN_LABELS = "train-labels-idx1-ubyte"
    TEST_IMAGES = "t10k-images-idx3-ubyte"
    TEST_LABELS = "t10k-labels-idx1-ubyte"

    @classmethod
    def download(cls, settype: str = "train"):
        if settype not in ["train", "test"]:
            raise ValueError(f"Invalid set type {settype}. Valid values are 'train' and 'test'")

        import requests
        import os

        def _download(url):
            response = requests.get(url)
            name = url.split("/")[-1]
            with open(name, "wb") as f:
                f.write(response.content)

            import gzip
            import shutil

            with gzip.open(name, "rb") as fIn:
                with open(name[:-3], "wb") as fOut:
                    shutil.copyfileobj(fIn, fOut)

        if settype == "train":
            if not os.path.exists(cls.TRAIN_IMAGES):
                _download(cls.URL + cls.TRAIN_IMAGES + ".gz")
            if not os.path.exists(cls.TRAIN_LABELS):
                _download(cls.URL + cls.TRAIN_LABELS + ".gz")
        elif settype == "test":
            if not os.path.exists(cls.TEST_IMAGES):
                _download(cls.URL + cls.TEST_IMAGES + ".gz")
            if not os.path.exists(cls.TEST_LABELS):
                _download(cls.URL + cls.TEST_LABELS + ".gz")

    @classmethod
    def get(cls, settype: str = "train"):
        if settype not in ["train", "test"]:
            raise ValueError(f"Invalid set type {settype}. Valid values are 'train' and 'test'")

        cls.download(settype)

        if settype == "train":
            images, labels = IDXFormatLoader.load(cls.TRAIN_IMAGES), IDXFormatLoader.load(cls.TRAIN_LABELS)
        elif settype == "test":
            images, labels = IDXFormatLoader.load(cls.TEST_IMAGES), IDXFormatLoader.load(cls.TEST_LABELS)
        
        return images, labels