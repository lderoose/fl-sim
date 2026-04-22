import time
from contextlib import contextmanager
from typing import Generator, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src import utils
from src.lib.stopping_criterion import EarlyStopping
from src.report.report import Report
from src.trainer.base_trainer import Trainer


class ClassicalTrainer(Trainer):
    """Performs the training procedure in a classical (non-federated) manner.

    A single model is trained on the full training set using standard mini-batch
    gradient descent.
    """

    def __init__(
        self,
        conf: dict,
        model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        report: Report,
    ) -> None:
        """Initialize the classical trainer.

        Args:
            conf (dict): Configuration dictionary produced by `ConfParser`.
            model (nn.Module): PyTorch model to train.
            x_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training labels.
            report (Report): Report instance used to record training metrics.
        """
        self.conf = conf
        self.device = conf["device"]
        self.model = model.to(self.device)
        self.x_train = x_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.report = report

        self.stopping_criterion: Optional[EarlyStopping] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
        return None

    @contextmanager
    def context_epoch(self, epoch: int) -> Generator[None, None, None]:
        """Context manager wrapping a single training epoch.

        Records execution time and fills the report after the epoch body runs.

        Args:
            epoch (int): Index of the current epoch.
        """
        start_time_epoch = time.time()

        yield

        # save relevant information
        self.model.eval()
        with torch.no_grad():
            self.report.fill(
                epoch=epoch,
                model=self.model,
                time_exec=utils.compute_exec_time(start=start_time_epoch),
            )

    def __call__(self) -> None:
        """Run the full classical training procedure and save the report."""
        should_stop = False
        criterion = self.conf.get("params_master")["criterion"]()

        opt_class = self.conf.get("params_master")["optimizer"]["name"]
        opt_kwargs = self.conf.get("params_master")["optimizer"]["kwargs"]
        optimizer = opt_class(self.model.parameters(), **opt_kwargs)

        # manage stopping
        sc_class = self.conf.get("params_master")["stopping_criterion"]["name"]
        if sc_class is not None:
            sc_kwargs = self.conf.get("params_master")["stopping_criterion"]["kwargs"]
            self.stopping_criterion = sc_class(**sc_kwargs)

        # manage lr_scheduler
        lr_class = self.conf.get("params_master")["lr_scheduler"]["name"]
        if lr_class is not None:
            lr_kw = self.conf.get("params_master")["lr_scheduler"]["kwargs"]
            self.lr_scheduler = lr_class(optimizer=optimizer, **lr_kw)

        trainset = TensorDataset(self.x_train, self.y_train)
        trainloader = DataLoader(
            trainset,
            batch_size=self.conf.get("params_master")["batch_size"],
            shuffle=False,
        )

        n_epochs = self.conf.get("params_master")["n_epochs"]

        for epoch in tqdm(range(n_epochs)):
            if not should_stop:
                with self.context_epoch(epoch):
                    self.model.train()
                    for batch in trainloader:
                        inputs, labels = batch

                        optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                # manage stopping & lr_scheduling
                ## compute loss on validation
                self.model.eval()
                with torch.no_grad():
                    if self.report.x_val is not None:
                        y_val_pred = self.model(self.report.x_val)
                        loss_val = criterion(y_val_pred, self.report.y_val)

                        if self.lr_scheduler is not None:
                            self.previous_lr = optimizer.param_groups[0]["lr"]
                            self.lr_scheduler.step(loss_val)
                            self.current_lr = optimizer.param_groups[0]["lr"]
                            if self.current_lr != self.previous_lr:
                                print(
                                    f"LR has been updated from {self.previous_lr} to {self.current_lr}"
                                )
                            self.report.fill(epoch=epoch, learning_rate=self.current_lr)

                        ## checks whether we have to stop
                        if self.stopping_criterion is not None:
                            self.stopping_criterion(
                                epoch_score=loss_val.item(), model=self.model
                            )
                            if self.stopping_criterion.early_stop:
                                should_stop = True

        self.report.save()
        return None
