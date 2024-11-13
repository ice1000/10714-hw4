import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        super().__init__(transforms)

        if train:
            data_files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            data_files = ['test_batch']

        self.X = []
        self.y = []

        for file_name in data_files:
            file_path = os.path.join(base_folder, file_name)
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                self.X.append(data_dict[b'data'])
                self.y.append(data_dict[b'labels'])

        self.X = np.concatenate(self.X).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = np.concatenate(self.y).astype(np.int64)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        image, label = self.X[index], self.y[index]
        if len(image.shape) > 1:
            image = np.array(list(map(self.apply_transforms, image)))
        else:
            image = self.apply_transforms(image)
        return image, label

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return len(self.X)
