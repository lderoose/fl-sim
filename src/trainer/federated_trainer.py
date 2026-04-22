import copy
import time
from contextlib import contextmanager
from itertools import chain
from typing import Generator, Optional

import numpy as np
from torch import nn
from tqdm import tqdm

from conf.convention import MASTER_ID
from src import utils
from src.communication.communicator import LocalCommunicator
from src.preprocessing.data_manager import DataManager
from src.report.report import Report
from src.trainer.base_trainer import Trainer
from src.user.master import Master
from src.user.worker import Worker


class FederatedTrainer(Trainer):
    """Performs the training procedure in a federated manner.

    Orchestrates a master-worker architecture where each worker trains on a
    local data shard and sends compressed or dense gradients to the master,
    which aggregates them and broadcasts updated weights back to the workers.
    """

    def __init__(
        self,
        conf: dict,
        master_model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        report: Report,
        dic_indexes: Optional[dict[str, list[int]]] = None,
    ) -> None:
        """Initialize the federated trainer.

        Creates the master, instantiates all workers, distributes data indices,
        computes per-step weights, and sets up the communicator.

        Args:
            conf (dict): Configuration dictionary produced by ConfParser.
            master_model (nn.Module): PyTorch model whose weights are shared with workers.
            x_train (np.ndarray): Full training input data.
            y_train (np.ndarray): Full training labels.
            report (Report): Report instance used to record training metrics.
            dic_indexes: Optional[dict[str, list[int]]]: for FEMNIST dataset. Default to None.
        """
        self.conf = conf
        self.master_model = master_model
        self.x_train = x_train
        self.y_train = y_train
        self.report = report

        # general infos -------
        self.n_epochs = self.conf.get("params_master")["n_epochs"]
        self.n_workers = self.conf.get("n_workers")
        self.n_layers = utils.compute_n_layers(model=master_model)
        self.n_steps = conf["params_master"]["n_steps"]

        # get data indexes for each worker
        if self.n_steps is not None:
            self.dic_idx_workers, global_batch_size = DataManager.get_idx_workers_A(
                n_datas=len(x_train),
                n_workers=conf.get("n_workers"),
                n_steps=self.n_steps,
            )
        else:
            global_batch_size = conf["params_master"]["batch_size"]
            self.dic_idx_workers = DataManager.get_idx_workers_B(
                n_datas=len(x_train),
                n_workers=conf.get("n_workers"),
                batch_size=global_batch_size,
            )
            self.n_steps = utils.compute_nsteps(self.dic_idx_workers)

        # at each step there is a weight attributed to the worker regarding data they owned for this step
        self.tensor_weights = utils.compute_weights(
            dic_idx_workers=self.dic_idx_workers,
            n_steps=self.n_steps,
            precision=self.conf["params_master"]["float_precision"],
        )

        # is there a compressor?
        self.is_compression, self.compressor_name = self._get_compressor_info()

        # is there a variant?
        self.is_variant, self.variant_name = self._get_variant_info()

        # is sparse weight updates enabled?
        self.use_sparse_weight_updates = self._get_sparse_weight_updates_info()

        # is there a stopping criterion?
        self.is_stopping_criterion = (
            True
            if self.conf["params_master"]["stopping_criterion"]["name"] is not None
            else False
        )

        # user creation -------
        self.communicator = LocalCommunicator(
            n_users=1 + self.n_workers
        )  # +1 for the master
        self.master = self._init_master(communicator=self.communicator)
        self.ls_workers = self._init_workers(communicator=self.communicator)
        self.ls_workers_id = [w.id for w in self.ls_workers]

        print(f"#datas in train: {len(x_train)}")
        print(f"#steps {self.n_steps}")
        print(f"#workers: {self.n_workers}")
        print(f"global batch size: {global_batch_size}")
        print(f"batch size: {len(self.dic_idx_workers['0'][0])}")

        return None

    def _init_master(self, communicator: LocalCommunicator) -> Master:
        """Create the master user and load the model onto it."""
        master = Master(
            id=MASTER_ID,
            communicator=communicator,
            device=self.conf["device"],
            params=self.conf.get("params_master"),
            n_workers=self.n_workers,
            n_layers=self.n_layers,
        )

        master.load_model(model=copy.deepcopy(self.master_model))
        return master

    def _init_workers(self, communicator: LocalCommunicator) -> list[Worker]:
        """Create all workers and equip each with its data shard, model copy, and guidelines."""
        ls_workers: list[Worker] = []
        for worker_id in range(self.n_workers):
            str_worker_id = str(worker_id)
            ls_idx = self.dic_idx_workers[str_worker_id]
            ls_idx_flatten = list(chain.from_iterable(ls_idx))

            worker: Worker = Worker(
                id=str_worker_id, communicator=communicator, device=self.conf["device"]
            )
            worker.load_data(
                data=self.x_train[ls_idx_flatten], target=self.y_train[ls_idx_flatten]
            )

            # we assume that each worker knows the model, which means that sending the architecture has no impact on communication costs
            worker.load_model(
                copy.deepcopy(self.master.model)
            )  # same weights init like master, but distinct object
            worker.load_guidelines(
                guidelines=self.conf.get("params_master"), n_layers=self.n_layers
            )
            ls_workers.append(worker)
        return ls_workers

    def _get_compressor_info(self) -> tuple[bool, str]:
        """Extract compressor information from the configuration.

        Returns:
            tuple[bool, str]: Whether compression is enabled, and the compressor class name.
        """
        is_compression = (
            True
            if self.conf["params_master"]["compressor"]["name"] is not None
            else False
        )
        compressor_name = (
            self.conf["params_master"]["compressor"]["name"].__name__
            if is_compression
            else ""
        )
        return is_compression, compressor_name

    def _get_variant_info(self) -> tuple[bool, str]:
        """Extract  variant information from the configuration.

        Returns:
            tuple[bool, str]: Whether a variant is enabled, and the variant class name.
        """
        is_variant = (
            True if self.conf["params_master"]["variant"]["name"] is not None else False
        )
        variant_name = (
            self.conf["params_master"]["variant"]["name"].__name__ if is_variant else ""
        )
        return is_variant, variant_name

    def _get_sparse_weight_updates_info(self) -> bool:
        """Extract the sparse weight updates flag from the configuration.

        Returns:
            bool: True if sparse weight updates are enabled.
        """
        return self.conf["params_master"]["sparse_weight_updates"]

    def actions_stop_training(
        self, report: Report, epoch: int, why: Optional[Exception]
    ) -> None:
        """Record stopping information in the report when training is interrupted early.

        Args:
            report (Report): Report instance to update.
            epoch (int): Epoch at which training stops (0-indexed).
            why (Optional[Exception]): Reason for stopping (e.g. "early_stopping").
        """
        # remember we stop and why
        report.report["params"]["params_master"]["special_behaviour"] = {
            "stop_training": why
        }
        # remember when we stop
        report.report["params"]["params_master"]["n_epochs_up"] = (
            epoch + 1
        )  # epoch loop starts at 0
        print(f"*** MET STOPPING CRITERION : {why} ***")
        return None

    def _worker_involvement(self, worker_id: int, step: int) -> tuple[bool, int]:
        """Determine whether a worker participates in a given step.

        A worker is involved if it has at least one data index assigned for that step.

        Args:
            worker_id (int): Identifier of the worker.
            step (int): Index of the current step.

        Returns:
            tuple[bool, int]: Whether the worker is involved, and the number of data
                points it processes at this step.
        """
        # init
        is_worker_involved = False

        try:
            ls_idx = self.dic_idx_workers[worker_id][step]
        except IndexError:
            ls_idx = []
        n_idx = len(ls_idx)

        if n_idx > 0:
            is_worker_involved = True

        return is_worker_involved, n_idx

    def _learning_rate_management(
        self, epoch: int, metric_name: str = "loss_validation"
    ) -> None:
        """Update the learning rate scheduler at the end of an epoch if one is configured."""
        if self.conf.get("params_master")["lr_scheduler"]["name"] is not None:
            lr = self.master.update_scheduler(
                loss=self.report.report["data_epoch"][epoch][metric_name]
            )
            self.report.fill(epoch=epoch, learning_rate=lr)
        return None

    def _stopping_criterion_management(self, epoch: int) -> None:
        """Check the stopping criterion at the end of an epoch and trigger early stopping if met."""
        if self.is_stopping_criterion:
            self.STOP_TRAINING = self.master.monitor_stopping(
                x_val=self.report.x_val, y_val=self.report.y_val
            )
            why = "early_stopping"

            # what should we do when we know we need to stop the training procedure
            if self.STOP_TRAINING:
                self.actions_stop_training(report=self.report, epoch=epoch, why=why)
        return None

    def _reset_worker_data_consumption(self) -> None:
        """Notify all workers to reset their data consumption counter for the next epoch."""
        for worker in self.ls_workers:
            worker.reset_data_consumption()
        return None

    def _first_epoch_management(self) -> None:
        """Perform initialization steps that must run exactly once before the first epoch.

        - Broadcasts the master's initial weights to every worker.
        - Records the planned number of epochs in the report.
        """
        # send the initial weights
        for worker_id, worker in enumerate(self.ls_workers):
            self.master.send_message(
                to=str(worker_id),
                subject="update_weights",
                data=self.master.model.state_dict(),
            )
            worker.receive_message()

        # for plots
        self.report.report["params"]["params_master"]["n_epochs_up"] = self.n_epochs
        return None

    @contextmanager
    def context_epoch(self, epoch: int) -> Generator[None, None, None]:
        """Context manager wrapping the body of a single training epoch.

        Measures execution time, checks the stopping criterion, resets worker
        data counters, and fills the report after the step loop completes.

        NB: using a context manager avoids deeply nested loops.

        Args:
            epoch (int): Index of the current epoch.
        """
        # starts measuring the execution time
        start_time_epoch = time.time()

        # enter the step loop
        yield
        # exit the step loop

        self._stopping_criterion_management(epoch)
        self._reset_worker_data_consumption()

        # save relevant information
        self.report.fill(
            epoch=epoch,
            model=self.master.model,
            time_exec=utils.compute_exec_time(start=start_time_epoch),
        )
        self.report.fill(
            epoch=epoch, dic_cc=self.communicator.cc_tracker.get_cc_summary()
        )

        self._learning_rate_management(epoch)

        self.pbar.update(1)

    @contextmanager
    def context_step(self, step: int) -> Generator[None, None, None]:
        """Context manager wrapping the body of a single gradient step.

        After the worker loop completes, triggers model aggregation, broadcasts
        updated weights to all workers, and resets collected worker results.

        NB: using a context manager avoids deeply nested loops.

        Args:
            step (int): Index of the current step within the epoch.
        """

        # enter the worker loop
        yield
        # exit the worker loop

        if not self.STOP_TRAINING:

            self.master.update_model(weights=self.tensor_weights[:, step])

            if self.is_compression and self.use_sparse_weight_updates:
                master_update = self.master.build_payload()
            else:
                master_update = self.master.model.state_dict()

            # update the weights of every worker
            for worker_id, worker in enumerate(self.ls_workers):
                self.master.send_message(
                    to=str(worker_id), subject="update_weights", data=master_update
                )
                worker.receive_message()

            # reset the worker results
            self.master.reset_worker_results()

        return None

    @contextmanager
    def context_worker(self, worker: Worker, step: int) -> Generator[None, None, None]:
        """Context manager wrapping the processing of a single worker at a given step.

        If the worker has data assigned for this step, the master instructs it to
        perform one gradient step and collects the result.

        NB: using a context manager avoids deeply nested loops.

        Args:
            worker (Worker): The worker to process.
            step (int): Index of the current step within the epoch.
        """
        # checks if the worker is involved in this step and for how many data
        is_worker_involved, n_datas_worker = self._worker_involvement(
            worker_id=worker.id, step=step
        )

        if is_worker_involved:
            self.master.send_message(
                to=worker.id,
                subject="perform_one_step",
                data={"n_datas": n_datas_worker},
            )
            worker.receive_message()  # the worker has automatically send a message to the master with the result of his calculation
            self.master.receive_message(from_=worker.id)

        yield

    def __call__(self) -> None:
        """Run the full federated training procedure and save the report."""
        self.STOP_TRAINING = (
            False  # updated by the stopping criterion at each epoch if needed
        )
        self.pbar = tqdm(total=self.n_epochs)

        self._first_epoch_management()

        for epoch in range(self.n_epochs):
            if self.STOP_TRAINING:
                break
            with self.context_epoch(epoch):
                for step in range(self.n_steps):
                    with self.context_step(step):
                        for worker in self.ls_workers:
                            with self.context_worker(worker=worker, step=step):
                                continue

        self.report.save()
        return None
