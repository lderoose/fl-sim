from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Unpack

from conf.convention import MASTER_ID
from src.lib import compressor, variant
from src.utils import AllStaticMethods


class CommunicationCostComputer(metaclass=AllStaticMethods):
    """Computes theoretical communication costs for a federated learning setup.

    All methods are static (enforced by the AllStaticMethods metaclass).
    Supports cost estimation with and without gradient compression and variants.
    """

    LS_HANDLED_PRECISION = ["float16", "float32", "float64"]
    LS_HANDLED_COMPRESSOR = [
        None,
        compressor.TopKCompressor,
        compressor.RandomKCompressor,
    ]
    LS_HANDLED_VARIANT = [None, variant.ErrorFeedback, variant.ProjFL]

    def _check_input_parameters(
        n_epochs: int,
        n_steps: int,
        n_workers: int,
        model: nn.Module,
        precision: str,
        cpr: Optional[dict] = None,
        vrt: Optional[dict] = None,
    ) -> None:
        """Validate the input parameters before computing the communication cost.

        Args:
            n_epochs (int): Number of training epochs (must be > 0).
            n_steps (int): Number of steps per epoch (must be > 0).
            n_workers (int): Number of workers (must be > 0).
            model (nn.Module): PyTorch model used for training.
            precision (str): Float precision string; must be in LS_HANDLED_PRECISION.
            cpr (Optional[dict]): Compressor configuration dict. If provided, must
                contain a "name" key from LS_HANDLED_COMPRESSOR.
            vrt (Optional[dict]): Variant configuration dict. If provided, must
                contain a "name" key from LS_HANDLED_VARIANT.

        Raises:
            AssertionError: If any parameter is invalid.
        """
        assert n_epochs > 0, f"expected parameter n_epochs > 0, get {n_epochs=} instead"
        assert n_steps > 0, f"expected parameter n_steps > 0, get {n_steps=} instead"
        assert (
            n_workers > 0
        ), f"expected parametern_workers > 0, get {n_workers=} instead"
        assert isinstance(
            model, nn.Module
        ), f"expected parameter model is a pytorch model, get '{type(model)}' instead"
        assert (
            precision in __class__.LS_HANDLED_PRECISION
        ), f"expected precision in {__class__.LS_HANDLED_PRECISION}, get {precision=} instead"
        if cpr is not None:
            assert isinstance(
                cpr, dict
            ), f"expected a dictionnary for parameter 'cpr', get {type(cpr)} instead"
            assert (
                cpr["name"] in __class__.LS_HANDLED_COMPRESSOR
            ), f"expected cpr in {__class__.LS_HANDLED_COMPRESSOR}, get {cpr['name']} instead"

        if vrt is not None:
            assert isinstance(
                vrt, dict
            ), f"expected a dictionnary for parameter 'vrt', get {type(vrt)} instead"
            assert (
                vrt["name"] in __class__.LS_HANDLED_VARIANT
            ), f"expected vrt in {__class__.LS_HANDLED_VARIANT}, get {vrt['name']} instead"
        return None

    def get_n_bytes_from_precision(precision: str) -> int:
        """Return the number of bytes used to represent a single scalar at the given precision.

        Args:
            precision (str): One of "float16", "float32", or "float64".

        Returns:
            int: Number of bytes per scalar (2, 4, or 8).

        Raises:
            NotImplementedError: If the precision string is not supported.
        """
        if precision == "float32":
            return 4
        elif precision == "float64":
            return 8
        elif precision == "float16":
            return 2
        else:
            raise NotImplementedError(f"this precision '{precision}' is not supported.")

    def compute_tensor_cc(tensor: torch.Tensor) -> int:
        """Compute the number of bytes needed to transmit a tensor.

        Counts only the element data (element_size * numel), not the Python
        object overhead.

        Args:
            tensor (torch.Tensor): Tensor whose transmission cost is computed.

        Returns:
            int: Number of bytes.
        """
        ms_one_element = (
            tensor.element_size()
        )  # according to the precision of the tensor
        n_elements = tensor.numel()
        ms_total = ms_one_element * n_elements  # number of bytes
        return ms_total

    def byte_to_gib(n_bytes: int, precision: int = 4) -> float:
        """Convert a byte count to gibibytes (GiB, not GB).

        Args:
            n_bytes (int): Number of bytes to convert.
            precision (int): Number of decimal places in the result.

        Returns:
            float: Value in GiB.
        """
        BYTES_PER_GIB = 1_073_741_824  # 1024 ** 3
        return round(n_bytes / BYTES_PER_GIB, precision)

    def count_parameters(model: nn.Module) -> int:
        """Count the number of trainable parameters in a PyTorch model.

        Args:
            model (nn.Module): The model to inspect.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_n_layers(model: nn.Module) -> int:
        """Count the number of parameter tensors in a PyTorch model.

        Args:
            model (nn.Module): The model to inspect.

        Returns:
            int: Number of parameter tensors.
        """
        # recall: model.parameters() is a generator
        return sum(1 for _ in model.parameters())

    def count_parameters_per_layer(model: nn.Module) -> dict:
        """Count the number of parameters for each layer of a PyTorch model.

        Args:
            model (nn.Module): The model to inspect.

        Returns:
            dict: Mapping from layer index to parameter count.
        """
        return {i: np.prod(p.shape) for i, p in enumerate(model.parameters())}

    def __call__(
        n_epochs: int,
        n_steps: int,
        n_workers: int,
        model: nn.Module,
        precision: str = "float32",
        **kwargs: Unpack[dict],
    ) -> int | dict:
        """Compute the theoretical communication cost for a federated training run.

        NB: n_steps is an integer, meaning all workers are assumed to perform the
        same number of steps per epoch.

        Args:
            n_epochs (int): Number of training epochs.
            n_steps (int): Number of gradient steps per epoch.
            n_workers (int): Number of workers.
            model (nn.Module): Model used for training.
            precision (str): Float precision. Defaults to "float32".
            one_epoch (bool, optional): If True, output the cost for one epoch only
                (divided by n_epochs). Defaults to False.
            detailed_output (bool, optional): If True, return a dict with total,
                master-to-workers, and workers-to-master costs. Defaults to False.
            cpr (dict, optional): Compressor configuration dict.
            vrt (dict, optional): Variant configuration dict.

        Raises:
            ValueError: If a variant is configured without a compressor.

        Returns:
            int | dict: Total communication cost in bytes, or a detailed dict
                with keys "total", "master_to_workers", and "workers_to_master".
        """

        # unpack kwargs
        cpr = kwargs.get("cpr", None)
        vrt = kwargs.get("vrt", None)
        one_epoch = kwargs.get("one_epoch", False)
        detailed_output = kwargs.get("detailed_output", False)

        __class__._check_input_parameters(
            n_epochs=n_epochs,
            n_steps=n_steps,
            n_workers=n_workers,
            model=model,
            precision=precision,
            cpr=cpr,
            vrt=vrt,
        )

        # init
        is_variant = False
        is_compression = False

        if cpr is not None:
            if cpr["name"] is not None:
                cpr_class = cpr["name"]
                is_compression = True

                # create an instance of the compressor to be able to compute the theoretical number of outputs
                cpr_kwargs = cpr["kwargs"]
                cpr_inst = cpr_class(**cpr_kwargs)

        if vrt is not None:
            if vrt["name"] is not None:
                vrt_class = vrt["name"]
                vrt_name = vrt_class.__name__
                is_variant = True

        if not is_compression and is_variant:
            raise ValueError(f"expected to have compression when variant is set")

        n_params_model = __class__.count_parameters(model)
        n_bytes_per_elem = __class__.get_n_bytes_from_precision(precision)
        master_to_workers = (
            n_epochs * n_steps * n_workers * n_params_model * n_bytes_per_elem
        )

        # compute the number of outputs from the workers to the master
        n_outputs_workers_to_master = 0

        if is_compression:

            for weight_layer in model.parameters():
                n_outputs_workers_to_master += cpr_inst.compute_th_n_outputs(
                    weight_layer
                )

                if is_variant:
                    if vrt_name == "ProjFL":
                        # alpha
                        n_outputs_workers_to_master += vrt["kwargs"]["K"]

            # multiply according to the precision
            workers_to_master = (
                n_epochs
                * n_steps
                * n_workers
                * n_outputs_workers_to_master
                * n_bytes_per_elem
            )

        else:
            workers_to_master = master_to_workers

        total = master_to_workers + workers_to_master

        # divide every potential output by the number of epochs
        if one_epoch:
            total /= n_epochs
            master_to_workers /= n_epochs
            workers_to_master /= n_epochs

        if detailed_output:
            dic_out = {
                "total": int(total),
                "master_to_workers": int(master_to_workers),
                "workers_to_master": int(workers_to_master),
            }
            return dic_out

        else:
            return total


class CommunicationCostTracker:
    """Track the actual communication cost incurred during training in a star network.

    Maintains per-user byte counters that are updated each time a message is sent.
    """

    def __init__(self, n_workers: int) -> None:
        """Initialize the communication cost tracker.

        Args:
            n_workers (int): Number of workers in the simulation.
        """
        self.n_workers = n_workers

        self.dic_cc = self._build_dic_cc()
        return None

    def _build_dic_cc(self) -> dict[str, dict[str, int] | int]:
        """Create the dictionary that stores the cumulative communication cost per user.

        The master entry maps to a sub-dict keyed by worker id (cost of messages
        sent from master to that worker). Each worker entry is a scalar (cost of
        messages sent from that worker to the master).

        Returns:
            dict: Initialized communication cost structure with all values at 0.
        """
        dic = {
            MASTER_ID: {f"{worker_id}": 0 for worker_id in range(self.n_workers)},
        }
        for worker_id in range(self.n_workers):
            dic[f"{worker_id}"] = 0
        return dic

    def update(self, src: str, to: str, x: int) -> None:
        """Add x bytes to the communication cost counter for the given sender.

        Args:
            src (str): Identifier of the message sender.
            to (str): Identifier of the message recipient.
            x (int): Number of bytes to add.

        Raises:
            ValueError: If a worker attempts to send directly to another worker.
        """
        # the master has sent a message
        if src == MASTER_ID:
            self.dic_cc[MASTER_ID][to] += x

        # a worker has sent a message to the master
        else:
            if to == MASTER_ID:
                self.dic_cc[src] += x
            else:
                raise ValueError(f"workers can't communicate together.")
        return None

    def get_cc_master_to_workers(self) -> int:
        """Return the total bytes sent from the master to all workers.

        Returns:
            int: Cumulative communication cost in bytes.
        """
        return sum(self.dic_cc[MASTER_ID].values())

    def get_cc_workers_to_master(self) -> int:
        """Return the total bytes sent from all workers to the master.

        Returns:
            int: Cumulative communication cost in bytes.
        """
        cc = 0
        for worker_id in range(self.n_workers):
            cc += self.dic_cc[str(worker_id)]
        return cc

    def get_cc(self) -> int:
        """Return the total bidirectional communication cost between master and workers.

        Returns:
            int: Sum of master-to-workers and workers-to-master costs in bytes.
        """
        cc_mtw = self.get_cc_master_to_workers()
        cc_wtm = self.get_cc_workers_to_master()
        return cc_mtw + cc_wtm

    def get_cc_summary(self) -> dict[str, int]:
        """Return a summary dict of all communication cost directions.

        Returns:
            dict[str, int]: Keys are "master_to_workers", "workers_to_master",
                and "total", with values in bytes.
        """
        return {
            "master_to_workers": self.get_cc_master_to_workers(),
            "workers_to_master": self.get_cc_workers_to_master(),
            "total": self.get_cc(),
        }
