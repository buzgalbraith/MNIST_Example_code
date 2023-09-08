## this whole file is just helper code to load and format the mnist dataset, dont worry to much about it.
## data is from here https://www.kaggle.com/datasets/hojjatk/mnist-dataset did not use the pytorch dataset due to mac bug with downloading
## the dataloader class is adapted (with some data processing and formatting changes) from this https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook

import numpy as np
import struct
from array import array
from os.path import join
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Union, Tuple


class MNIST_DataObject(Dataset):
    """MNIST dataset object for pytorch dataloader"""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath: str,
        training_labels_filepath: str,
        test_images_filepath: str,
        test_labels_filepath: str,
        batch_size=1,
    ) -> None:
        """dataloader for mnist dataset.

        Args:
            training_images_filepath (str): path to training images
            training_labels_filepath (str): path to training labels
            test_images_filepath (str): path to test images
            test_labels_filepath (str): path to test labels
            batch_size (int, optional): batch size for dataloader. Defaults to 1.
        """
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
        self.batch_size = batch_size

    def read_images_labels(
        self, images_filepath: str, labels_filepath: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """read images and labels from filepath.
        Args:
            images_filepath (str): path to images
            labels_filepath (str): path to labels
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of images and labels
        """
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = torch.zeros(len(labels), rows, cols, 1)
        for i in range(size):
            img = torch.Tensor(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28, 1)
            images[i][:] = img
        return images, torch.Tensor(labels)

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """load data from filepaths.

        Returns:
            Tuple[DataLoader, DataLoader]: tuple of train and test dataloaders
        """
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        train_object = MNIST_DataObject(x=x_train, y=y_train)
        test_object = MNIST_DataObject(x=x_test, y=y_test)
        return DataLoader(train_object, batch_size=self.batch_size), DataLoader(
            test_object, batch_size=self.batch_size
        )


## setting these variables here so we dont have to pass them around
input_path = "MNIST_data/"
training_images_filepath = join(
    input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
)
training_labels_filepath = join(
    input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
test_images_filepath = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")
