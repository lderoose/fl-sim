from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from conf.paths import DATA_DIR


def load_and_preprocess_cifar10(path_data: Path = DATA_DIR) -> tuple[torch.Tensor, ...]:
    """Load and preprocess the CIFAR-10 dataset.

    Converts to float32, permutes HWC to CHW, and applies per-channel
    normalization. Assumes each worker applies the same preprocessing.

    Args:
        path_data (Path): Root directory for dataset storage. Defaults to DATA_DIR.

    Returns:
        tuple: (x_train, y_train, x_test, y_test).
    """
    trainset = CIFAR10(root=path_data, train=True, download=True)
    testset = CIFAR10(root=path_data, train=False, download=True)

    x_train = trainset.data.astype(np.float32) / 255.0
    x_test = testset.data.astype(np.float32) / 255.0

    # Permutation HWC -> CHW
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    # Normalization CIFAR-10 (per channel)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    y_train = np.array(trainset.targets)
    y_test = np.array(testset.targets)

    # convert to torch
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long().reshape(-1)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long().reshape(-1)

    return x_train, y_train, x_test, y_test
