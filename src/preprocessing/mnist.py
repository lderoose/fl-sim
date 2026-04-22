from pathlib import Path

import torch
from torchvision.datasets import MNIST

from conf.paths import DATA_DIR


def load_and_preprocess_mnist(path_data: Path = DATA_DIR) -> tuple[torch.Tensor, ...]:
    """Load and preprocess the MNIST dataset.

    Normalizes pixel values from [0, 255] to [-1, 1]. Assumes each worker
    applies the same preprocessing.

    Args:
        path_data (Path): Root directory for dataset storage. Defaults to DATA_DIR.

    Returns:
        tuple: (x_train, y_train, x_test, y_test).
    """
    trainset = MNIST(root=path_data, train=True, download=True)
    testset = MNIST(root=path_data, train=False, download=True)
    x_train = (
        (trainset.data / 255) - 0.5
    ) / 0.5  # ==> [0:255] --> [0:1] --> [-1:1]
    x_test = ((testset.data / 255) - 0.5) / 0.5  # ==> [0:255] --> [0:1] --> [-1:1]

    return x_train, trainset.targets, x_test, testset.targets
