## this file defines the our convolutional neural network class and some helper functions for validating results
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union


class conv_net(nn.Module):

    """simple convolutional net for mnist dataset."""

    def __init__(self, input_size: int = 28, output_size: int = 10) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size * input_size, 25 * 25)
        self.conv_layer = nn.Conv1d(25, 20, kernel_size=6, stride=1, padding=0)
        self.pool_layer = nn.MaxPool2d(kernel_size=13, stride=1, padding=1)
        self.output_layer = nn.Linear(in_features=10 * 10, out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv net forward pass.
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, x_dim, y_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, 9) corresponding to the 10 classes of the mnist dataset
        """
        x = x.reshape(x.shape[0], -1)  ## flatten the input
        x = self.input_layer(x)
        x = x.reshape(x.shape[0], 25, 25)  ## reshape to 25x25
        x = self.conv_layer(x)
        x = self.pool_layer(x)
        x = x.reshape(x.shape[0], -1)  ## flatten for output layer
        x = self.output_layer(x)
        return x

    def save_model(self, file_path: str) -> None:
        """save model to file_path."""
        torch.save(self.state_dict(), str(time.time()) + "_" + file_path)


## helper function to show images
def show_images(
    loader: DataLoader,
    model: Union[torch.nn.Module, None] = None,
    num_batches: int = 1,
    batch_size: int = 1,
) -> None:
    """shows images from the loader and optionally what a model predicted for those images.
    Args:
    loader (DataLoader): dataloader object
    model (torch.nn.Module, optional): model to predict with Defaults to None which will just show the true label.
    batch_size (int, optional): number of images to show per batch. Defaults to 1.
    num_batches (int, optional): number of batches to show. Defaults to 1.
    """
    fig, ax = plt.subplots(ncols=batch_size, nrows=num_batches)
    index = 0
    for x, y in loader:
        if index >= num_batches:
            break
        if model is not None:
            predicted_probabilities = model(x)
            predicted_label = torch.argmax(predicted_probabilities, dim=1)
        for j in range(batch_size):
            image = x[j].reshape(28, 28)
            predicted_label_test = (
                ", Predicted Label:" + str(predicted_label[j])[-2]
                if model is not None
                else ""
            )
            title_text = "True Label:" + str((y[j]))[-3] + predicted_label_test
            ax[index][j].imshow(image, cmap=plt.cm.gray)
            ax[index][j].set_title(title_text, fontsize=15)
        index += 1
    title_text = (
        "Examples from the MNIST Dataset"
        if model is None
        else "Predictions on the Test Dataset"
    )
    plt.suptitle(title_text)
    plt.show()


def plot_confusion_matrix(model: nn.Module, loader: DataLoader) -> None:
    """plots a confusion matrix for the model on the loader data.
    Args:
        model (nn.Module): model to predict with
        loader (DataLoader): dataloader object
        batch_size (int, optional): number of images to show per batch. Defaults to 1.
    """
    confusion_matrix = torch.zeros((10, 10))
    for x, y in loader:
        predicted_probabilities = model(x)
        predicted_label = torch.argmax(predicted_probabilities, dim=1)
        for i in range(x.shape[0]):
            confusion_matrix[int(y[i]), int(predicted_label[i])] += 1
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.set_xticks(torch.arange(10))
    ax.set_yticks(torch.arange(10))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            text = ax.text(
                j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="w"
            )
    fig.tight_layout()
    plt.show()
