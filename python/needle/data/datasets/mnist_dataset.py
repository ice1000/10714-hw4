import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def parse_mnist(image_filename, label_filename):
        """Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format

        Returns:
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                    data.  The dimensionality of the data should be
                    (num_examples x input_dim) where 'input_dim' is the full
                    dimension of the data, e.g., since MNIST images are 28x28, it
                    will be 784.  Values should be of type np.float32, and the data
                    should be normalized to have a minimum value of 0.0 and a
                    maximum value of 1.0.

                y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.int8 and
                    for MNIST will contain the values 0-9.
        """
        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            size = rows * cols
            X = np.frombuffer(f.read(), dtype=np.uint8, count=num_images * size).reshape(num_images, rows, cols, 1)
            X = X.astype(np.float32) / 255.0
        with gzip.open(label_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8, count=num_labels)
        return X, y

    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        self.images, self.labels = MNISTDataset.parse_mnist(image_filename, label_filename)

    def __getitem__(self, index) -> object:
        image, label = self.images[index], self.labels[index]
        if len(image.shape) > 1:
            image = np.array(list(map(self.apply_transforms, image)))
        else:
            image = self.apply_transforms(image)
        return image, label

    def apply_transforms(self, image):
        if self.transforms is not None:
            for tform in self.transforms:
                image = tform(image)
        return image

    def __len__(self) -> int:
        return len(self.images)
