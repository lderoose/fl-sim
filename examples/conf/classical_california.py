#!/usr/bin/env python3
"""Example: classical (non-federated) training without compression on California Housing."""
from typing import Optional

import torch.nn as nn
from torch import optim

from conf.paths import REPORT_DIR
from src import utils
from src.models import california_models
from conf.criterion import CONF_DATASET_CRITERION
from src.preprocessing.california import load_and_preprocess_california

DEVICE: str = "cpu"  # "cuda" or "cpu"
FEDERATED: bool = False
SEED: int = 2026
N_WORKERS: int = 1  # unused in classical mode
DATASET_NAME: str = "california"
PREPROCESSOR = load_and_preprocess_california

LIMIT_N_DATA_TRAIN: int | float | None = None  # use full dataset
LIMIT_N_DATA_TEST: int | float | None = None
VALIDATION_SIZE: Optional[float] = 0.2
BATCH_SIZE: int = 1096

MODEL: nn = california_models.CaliforniaModelSimple
FLOAT_PRECISION: str = "float32"  # "float16", "float32", "float64"

N_EPOCHS: int = 10
N_STEPS: Optional[int] = None  # derived from BATCH_SIZE

OPTIMIZER = {
    "name": optim.SGD,
    "kwargs": {"lr": 0.1, "weight_decay": 0},
}

# no compression
COMPRESSOR = {
    "name": None,
    "kwargs": {"k": 0.1, "k_as_factor": True},
}

# no variance reduction variant
VARIANT = {
    "name": None,
    "kwargs": {"alpha": 0.75, "beta": 0.9, "K": 1, "EF": False},
}

STOPPING_CRITERION = {
    "name": None,
    "kwargs": {
        "patience": 10,
        "mode": "min",
        "delta": 1e-4,
        "verbose": True,
        "path": "model_checkpoints/checkpoint.pt",
    },
}

# no effect without compression
SPARSE_WEIGHT_UPDATES: bool = False

CRITERION: nn = CONF_DATASET_CRITERION[DATASET_NAME]

LR_SCHEDULER = {
    "name": None,
    "kwargs": {"patience": 2, "factor": 0.1, "min_lr": 1e-3},
}

REPORT_PARAMS = {
    "path_save": str(
        REPORT_DIR / utils.get_report_name(cpr=COMPRESSOR, federated=FEDERATED)
    ),
    "loss_train": True,
    "loss_test": True,
    "loss_validation": True,
    "accuracy_train": False,  # regression task: no accuracy metric
    "accuracy_validation": False,
    "accuracy_test": False,
    "var_loss_train": False,
    "var_loss_validation": False,
    "var_loss_test": False,
    "time_exec": True,
    "communication_cost": False,  # no communication in classical mode
    "weights_l1_norm": False,
    "gradients_l1_norm": False,
    "gradients_l2_norm": False,
    "weights_l2_norm": False,
    "weights": False,
}
