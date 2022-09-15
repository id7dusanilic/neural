import os
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
    def download(cls, settype: str = "train", loc: str = "."):
        if settype not in ["train", "test"]:
            raise ValueError(f"Invalid set type {settype}. Valid values are 'train' and 'test'")

        import requests

        def _download(url, loc):
            response = requests.get(url)
            name = url.split("/")[-1]
            with open(os.path.join(loc, name), "wb") as f:
                f.write(response.content)

            import gzip
            import shutil

            with gzip.open(os.path.join(loc, name), "rb") as fIn:
                with open(os.path.join(loc, name[:-3]), "wb") as fOut:
                    shutil.copyfileobj(fIn, fOut)

        os.makedirs(loc, exist_ok=True)
        if settype == "train":
            if not os.path.exists(os.path.join(loc, cls.TRAIN_IMAGES)):
                _download(cls.URL + cls.TRAIN_IMAGES + ".gz", loc)
            if not os.path.exists(os.path.join(loc, cls.TRAIN_LABELS)):
                _download(cls.URL + cls.TRAIN_LABELS + ".gz", loc)
        elif settype == "test":
            if not os.path.exists(os.path.join(loc, cls.TEST_IMAGES)):
                _download(cls.URL + cls.TEST_IMAGES + ".gz", loc)
            if not os.path.exists(os.path.join(loc, cls.TEST_LABELS)):
                _download(cls.URL + cls.TEST_LABELS + ".gz", loc)

    @classmethod
    def get(cls, settype: str = "train", loc: str = "."):
        if settype not in ["train", "test"]:
            raise ValueError(f"Invalid set type {settype}. Valid values are 'train' and 'test'")

        loc = os.path.abspath(loc)
        cls.download(settype, loc)

        if settype == "train":
            imgLoc, labelLoc = os.path.join(loc, cls.TRAIN_IMAGES), os.path.join(loc, cls.TRAIN_LABELS)
        elif settype == "test":
            imgLoc, labelLoc = os.path.join(loc, cls.TEST_IMAGES), os.path.join(loc, cls.TEST_LABELS)

        images, labels = IDXFormatLoader.load(imgLoc), IDXFormatLoader.load(labelLoc)

        return images, labels


class CIFAR10:

    URL = "https://www.cs.toronto.edu/~kriz/"
    NAME = "cifar-10-python.tar.gz"
    DIRNAME = "cifar-10-python/cifar-10-batches-py"

    @classmethod
    def download(cls, loc: str = "."):
        import requests
        import tarfile

        if not os.path.exists(os.path.join(loc, cls.NAME)):
            url = cls.URL + cls.NAME
            response = requests.get(url)
            with open(os.path.join(loc, cls.NAME), "wb") as f:
                f.write(response.content)

            with tarfile.open(os.path.join(loc, cls.NAME)) as f:
                f.extractall(os.path.join(loc, cls.NAME[:-7]))

    @classmethod
    def get(cls, batch: str = "test", loc: str = "."):
        if batch not in ["test", 1, 2, 3, 4, 5]:
            raise ValueError(f"Invalid batch type {batch}. Valid values are 'train' and 1,2,3,4,5")

        loc = os.path.abspath(loc)
        cls.download(loc)

        def _unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
            return data

        filename = "test_batch" if batch == "test" else f"data_batch_{batch}"
        data = _unpickle(os.path.join(loc, cls.DIRNAME, filename))
        images = data[b"data"]
        labels = data[b"labels"]

        return images, labels
