#!/usr/bin/env python


import torch.nn as nn


CONF_DATASET_CRITERION : dict = {
    "femnist": nn.CrossEntropyLoss,
    "mnist": nn.CrossEntropyLoss,
    "california": nn.MSELoss,
    "cifar10": nn.CrossEntropyLoss,
}
