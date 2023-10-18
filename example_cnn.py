## this file defines the our convolutional neural network class and some helper functions for validating results
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union
import os
import glob


class conv_net(nn.Module):

    """simple convolutional net for mnist dataset."""

    def __init__(self, output_size: int = 10, hidden_size: int = 64) -> None:
        super().__init__()
        print("using hidden size {0}".format(hidden_size))
        self.conv_layer_1 = nn.Conv2d(in_channels = 1, out_channels=hidden_size, kernel_size=3, stride=1, padding=0) ## output size = [batch_size, output channels, height, width]
        self.pool_layer_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) ## output size = [batch_size, output channels, 10, 10]
        self.conv_layer_2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=0) ## output size = [batch_size, output channels, 6, 6]
        self.pool_layer_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.downsample = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=1, stride=5, padding=0) ## output size = [batch_size, output channels, 28, 28]
        self.fc_layer = nn.Linear(in_features =  6*6*hidden_size, out_features=64) ## output size = [batch_size, output_size]
        self.output_layer = nn.Linear(in_features =  64, out_features=10) ## output size = [batch_size, output_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv net forward pass.
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, x_dim, y_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, 9) corresponding to the 10 classes of the mnist dataset
        """
        x = x.reshape(x.shape[0], 1, 28, 28) ## reshape to batch size, channel in , height, width
        c1 = self.conv_layer_1(x) ## [batch size, channel, height , width]
        r1 = torch.nn.functional.relu(c1)
        p1 = self.pool_layer_1(r1) ## [batch size, channel, height , width
        ## skip connection 

        c2 = self.conv_layer_2(p1) 
        r2 = torch.nn.functional.relu(c2)
        p2 = self.pool_layer_2(r2)
        x3 = p2 + self.downsample(x)
        x3 = torch.nn.functional.avg_pool2d(x3, kernel_size=3, stride=2, padding=1)
        l1 = p2.reshape(x3.shape[0], -1)  ## flatten the output of the conv layers
        fc1 = self.fc_layer(l1)
        r3 = torch.nn.functional.relu(fc1)
        fc2 = self.output_layer(r3)
        return torch.nn.functional.softmax(fc2, dim=1) ## softmax to get probabilities

    def save_model(self, file_path: str) -> None:
        """save model to file_path.
        Args:
            file_path (str): path to save model to
        """
        ## format the file path
        save_path = "saved_models/" + file_path
        save_path = save_path if save_path[-3:] == ".pt" else save_path + ".pt"
        save_path = (
            save_path[:-3] + "_" + str(round(time.time())) + ".pt"
        )  ## rounding to take out miliseconds since the period could mess with the file path
        ## saving the model to the file.
        torch.save(self.state_dict(), save_path)


## helper function to show images
def show_images(
    loader: DataLoader,
    model: Union[torch.nn.Module, None] = None,
    num_batches: int = 1,
    batch_size: int = 1,
    save_fig: bool = True,
    show_fig: bool = False,
) -> None:
    """shows images from the loader and optionally what a model predicted for those images.
    Args:
    loader (DataLoader): dataloader object
    model (torch.nn.Module, optional): model to predict with Defaults to None which will just show the true label.
    batch_size (int, optional): number of images to show per batch. Defaults to 1.
    num_batches (int, optional): number of batches to show. Defaults to 1.
    save_fig (bool, optional): whether to save the figure. Defaults to True.
    show_fig (bool, optional): whether to show the figure. Defaults to False.
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
            ax[index][j].set_title(title_text, fontsize=7.5)
        index += 1
    title_text = (
        "Examples from the MNIST Dataset"
        if model is None
        else "Predictions on the Test Datase"
    )
    plt.suptitle(title_text)
    save_title = "mnist_example.png" if model is None else "mnist_predictions.png"
    ## saves the figure to a set location.
    if save_fig:
        plt.savefig("saved_figs/" + save_title)
    ## shows the figure (does not work on remote servers)
    if show_fig:
        plt.show()


def plot_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    save_fig: bool = True,
    show_fig: bool = False,
) -> None:
    """plots a confusion matrix for the model on the loader data.
    Args:
        model (nn.Module): model to predict with
        loader (DataLoader): dataloader object
        save_fig (bool, optional): whether to save the figure. Defaults to True.
        show_fig (bool, optional): whether to show the figure. Defaults to False.
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
    if save_fig:
        plt.savefig("saved_figs/confusion_matrix.png")
    if show_fig:
        plt.show()


def get_most_recent_model(file_path: str = "saved_models") -> str:
    """gets the most recent file in a directory.
    Args:
        file_path (str, optional): path to directory. Defaults to 'saved_models'.
    Returns:
        str: file path to most recent model
    """
    file_path = file_path if file_path[-1] == "/" else file_path + "/"
    list_of_files = glob.glob(
        file_path + "/*"
    )  # * means all if need specific format then *.csv
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
    except:
        raise ValueError(
            "Directory is empty, be sure to run cnn_train.py first and you are pointing to the correct directory"
        )
    return latest_file
