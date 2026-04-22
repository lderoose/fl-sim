import os

from typing import Optional

import torch.nn as nn

from torch import optim
from torch.optim import lr_scheduler

from conf.paths import REPORT_DIR
from src import utils
from src.lib import variant
from src.lib import compressor
from src.models import mnist_models
from src.models import cifar10_models
from src.models import california_models
from src.models import femnist_models
from conf.criterion import CONF_DATASET_CRITERION
from src.lib.stopping_criterion import EarlyStopping
from src.preprocessing.mnist import load_and_preprocess_mnist


# use gpu or cpu
DEVICE: str = "cpu"  # cuda or cpu

# use federate learning or not
FEDERATED: bool = True

# fix random
SEED: int = 2026

# how many workers - only used if FEDERATED == True
N_WORKERS: int = 3

# the dataset to use
DATASET_NAME: str = (
    "mnist"
)
PREPROCESSOR = load_and_preprocess_mnist

# reduce the size of the dataset
LIMIT_N_DATA_TRAIN: Optional[int | float] = (
    2000 * N_WORKERS if DATASET_NAME in ["mnist", "cifar10"] else 1.0
)  # for the entire training not only for a single worker
LIMIT_N_DATA_TEST: Optional[int | float] = (
    2000 * N_WORKERS if DATASET_NAME in ["mnist", "cifar10"] else 1.0
)  # for the entire training not only for a single worker
VALIDATION_SIZE: Optional[float] = 0.2  # a subset of the train dataset
BATCH_SIZE = None

MODEL: nn = mnist_models.MnistModelSimple

FLOAT_PRECISION: str = "float32"  # "float16", "float32", "float64"

# master
N_EPOCHS: int = 20
N_STEPS: Optional[int] = (
    10  # set this parameter to None involves using the function src.lib.preprocessing.data_manager.get_idx_workers_B
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
    "kwargs": {"k": 0.1, "k_as_factor": True},
}

VARIANT = {
    "name": None,  # variant.ErrorFeedback | variant.ProjFL | variant.ErrorFeedback21 | Diana | None
    "kwargs": {
        "alpha": 0.75,  # only for variant.ErrorFeedback
        "beta": 0.9,  # only for Diana
        "K": 1,  # only for variant.ProjFL
        "EF": False,  # only for variant.ProjFL
    },
}

STOPPING_CRITERION = {
    "name": EarlyStopping,  # None or EarlyStopping
    "kwargs": {
        "patience": 10,
        "mode": "min",
        "delta": 1e-4,
        "verbose": True,
        "path": "model_checkpoints/checkpoint.pt",
    },
}

# when compression is enabled, only send the modified weight indices to workers instead of the full model state
# set to False to always send the full model state to workers (required when compression is disabled)
SPARSE_WEIGHT_UPDATES: bool = True

# criterion choose the loss function according to the dataset
CRITERION: nn = CONF_DATASET_CRITERION[DATASET_NAME]  # see criterion.py

# learning rate scheduler (case federated with compression only)
LR_SCHEDULER = {
    "name": lr_scheduler.ReduceLROnPlateau,  # lr_scheduler.ReduceLROnPlateau, None
    "kwargs": {"patience": 2, "factor": 0.1, "min_lr": 1e-3},
}

# which informations should be saved during the training procedure
REPORT_PARAMS = {
    "path_save": os.path.join(
        REPORT_DIR, utils.get_report_name(cpr=COMPRESSOR, federated=FEDERATED)
    ),
    "loss_train": True,
    "loss_test": True,
    "loss_validation": True,
    "accuracy_train": True,
    "accuracy_validation": True,
    "accuracy_test": True,
    "var_loss_train": True,
    "var_loss_validation": True,
    "var_loss_test": True,
    "time_exec": True,
    "communication_cost": True,
    "weights_l1_norm": False,
    "gradients_l1_norm": False,
    "gradients_l2_norm": True,
    "weights_l1_norm": False,
    "weights_l2_norm": False,
    "weights": False,  # beware of file size
}
