import os

from typing import Optional

import torch.nn as nn

from torch import optim
from torch.optim import lr_scheduler

from src import utils
from src.lib import variant
from src.lib import compressor
from src.models import mnist_models
from src.models import cifar10_models
from src.models import california_models
from conf.criterion import CONF_DATASET_CRITERION
from src.lib.stopping_criterion import EarlyStopping
from src.preprocessing.california import load_and_preprocess_california


# use gpu or cpu
DEVICE: str = "cpu"  # cuda or cpu

# use federate learning or not
FEDERATED: bool = True

# fix random
SEED: int = 2025

# how many workers - only used if FEDERATED == True
N_WORKERS: int = 3

# the dataset to use
DATASET_NAME: str = (
    "california"
)
PREPROCESSOR = load_and_preprocess_california

# reduce the size of the dataset
LIMIT_N_DATA_TRAIN: int | float | None = (
    None  # for the entire training not only for a single worker
)
LIMIT_N_DATA_TEST: int | float | None = (
    None  # for the entire training not only for a single worker
)
VALIDATION_SIZE: float | None = 0.2  # a subset of the train dataset

MODEL: nn = california_models.CaliforniaModelSimple

FLOAT_PRECISION: str = "float32"  # "float16", "float32", "float64"

# master
N_EPOCHS: int = 10
N_STEPS: Optional[int] = (
    None  # set this parameter to None involves using the function src.lib.preprocessing.data_manager.get_idx_workers_B
)
BATCH_SIZE: int = (
    1096  # if federated learning means the batch_size for each worker is ~BATCH_SIZE // N_WORKERS
)

OPTIMIZER = {
    "name": optim.SGD,
    "kwargs": {
        "lr": 0.1,
        "weight_decay": 0,  # L2 regularization, None or 0 means deactivate
    },
}

# assume the master indicates the function for performing the sparsification to the worker
COMPRESSOR = {
    "name": None,  # None means no compression  [compressor.TopKCompressor, compressor.RandomKCompressor, None]
    "kwargs": {"k": 0.01, "k_as_factor": True},
}

VARIANT = {
    "name": None,  # variant.ErrorFeedback | variant.ProjFL | None
    "kwargs": {
        "alpha": 0.75,  # only for variant.ErrorFeedback
        "K": 3,  # only for variant.ProjFL
        "EF": True,  # only for variant.ProjFL
    },
}

STOPPING_CRITERION = {
    "name": None,  # None or EarlyStopping
    "kwargs": {
        "patience": 2,
        "mode": "min",
        "delta": 0.0,
        "verbose": True,
        "path": "model_checkpoints/checkpoint.pt",
    },
}

SPARSE_WEIGHT_UPDATES: bool = True

# criterion choose the loss function according to the dataset
CRITERION: nn = CONF_DATASET_CRITERION[DATASET_NAME]  # manages in conf/criterion.py

# learning rate scheduler (case federated with compression only)
LR_SCHEDULER = {
    "name": None,  # lr_scheduler.ReduceLROnPlateau | None
    "kwargs": {"patience": 50, "factor": 0.97},
}

# which informations should be saved during the training procedure
REPORT_PARAMS = {
    "path_save": "tmp/swu_enable.json",
    "loss_train": True,
    "loss_test": True,
    "loss_validation": True,
    "time_exec": True,
    "communication_cost": True,
    "weights_l1_norm": False,
    "gradients_l1_norm": False,
    "weights": False,  # beware of file size
}
