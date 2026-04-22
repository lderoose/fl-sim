import math
import os
from abc import ABC
from typing import Callable

import numpy as np
import torch
from torch import nn


class StoppingCriterion(ABC):
    """Abstract base class for training stopping criteria."""


class EarlyStopping(StoppingCriterion):
    """Stops training when a monitored metric has stopped improving.

    Tracks the best observed score and increments a patience counter each
    time the score does not improve by at least delta. Training is flagged
    to stop once the counter reaches the patience threshold. Supports both
    minimization (mode="min") and maximization (mode="max") objectives.
    """

    def __init__(
        self,
        patience: int = 2,
        mode: str = "min",
        delta: float = 0.0,
        verbose: bool = False,
        trace_func: Callable = print,
        path: str = "checkpoint.pt",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience (int): Number of epochs with no improvement after which
                training will be stopped.
            mode (str): One of "min" or "max". In "min" mode, training stops
                when the metric has stopped decreasing; in "max" mode, when it
                has stopped increasing.
            delta (float): Minimum change in the monitored metric to qualify
                as an improvement.
            verbose (bool): If True, prints a message each time the metric
                improves and a checkpoint is saved.
            trace_func (Callable): Function used for logging messages.
            path (str): File path for saving the model checkpoint.
        """
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        self.trace_func = trace_func
        self.path = path

        if self.mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

        # create the directory if not exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        return None

    def __call__(self, epoch_score: float, model: nn.Module) -> None:
        """Evaluate the current epoch score and update the stopping state.

        Sets self.early_stop to True when the patience counter reaches patience.
        NaN scores are treated as non-improvements and also increment the counter.

        Args:
            epoch_score (float): Monitored metric value for the current epoch.
            model (nn.Module): Current model; saved to disk if the score improves.
        """
        if math.isnan(epoch_score) or np.isnan(epoch_score):
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            if self.mode == "min":
                score = -1.0 * epoch_score

            elif self.mode == "max":
                score = epoch_score

            # first epoch
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(epoch_score, model)

            # best score NOT modified
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
                if self.counter >= self.patience:
                    self.early_stop = True

            # best score modified
            else:
                self.best_score = score
                self.save_checkpoint(epoch_score, model)
                self.counter = 0  # reset
        return None

    def save_checkpoint(self, epoch_score: float, model: nn.Module) -> None:
        """Save the model state dict to disk when the metric improves.

        Args:
            epoch_score (float): Current metric value (used for verbose logging).
            model (nn.Module): Model whose state dict is saved.
        """
        if self.verbose:
            self.trace_func(
                f"Validation metric moving : ({self.val_score:.6f} --> {epoch_score:.6f}).  Saving model ..."
            )

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            torch.save(model.state_dict(), self.path)

        self.val_score = epoch_score
        return None
