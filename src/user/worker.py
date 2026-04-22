from typing import Optional, OrderedDict

import torch
from torch import nn

from conf.convention import MASTER_ID
from src import utils
from src.communication.communicator import Communicator
from src.communication.message import Message
from src.user.user import User


class Worker(User):
    """A federated learning participant that trains the model on a local data shard.

    The Worker receives instructions from the Master (number of data points to process,
    updated model weights), performs local gradient computation, and sends the
    compressed or dense gradients back to the Master.
    """

    def __init__(
        self, id: str, communicator: Communicator, device: str, log_level=None
    ) -> None:
        """Initialize a worker.

        Args:
            id (str): Unique string identifier for this worker.
            communicator (Communicator): Shared communicator instance.
            device (str): Compute device string (e.g. "cpu" or "cuda:x").
            log_level: Python logging level. If None, logging is disabled.
        """
        super().__init__(id, communicator, device, log_level)

        self.data: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None
        self.n_datas: int = 0
        self.guidelines: Optional[dict] = None

        self.idx_start: int = 0
        self.idx_end: int = 0
        self.all_data_consumed: bool = False
        self.criterion: Optional[nn.Module] = None
        self.compressor: Optional[dict] = None
        self.sparse_weight_updates: Optional[bool] = None
        self.variant: Optional[dict] = None
        self.variant_name: Optional[str] = None
        self.is_variant: bool = False
        self.is_compression: bool = False

        self.first_weight_loading: bool = (
            True  # the first call is different because the master must systematically send its weights in uncompressed format.
        )
        return None

    def load_model(self, model: nn.Module) -> None:
        """Load the model and set it to training mode.

        Overrides User.load_model: workers always train, they never evaluate.

        Args:
            model (nn.Module): PyTorch model to load.
        """
        super().load_model(model)
        self.model.train()
        return None

    def receive_message(self) -> None:
        """Retrieve and dispatch the next message from the Master.

        Supported subjects:
            - "perform_one_step": run one gradient step and send results to master.
            - "update_weights": update local model weights from the master payload.
            - "test": used in pytest only.
        """
        msg: Message = super().receive_message()

        # pytest
        if msg.subject == "test":
            return msg

        elif msg.subject == "perform_one_step":
            dic_results = self._task_perform_one_step(n_datas=msg.data["n_datas"])

            # a message is created to send the results to the master
            self.send_message(to=MASTER_ID, subject="one_step_result", data=dic_results)

        elif msg.subject == "update_weights":
            self._task_update_weights_from_master(msg.data)

        else:
            raise NotImplementedError(f"{msg.subject} feature not implemented")
        return None

    def _update_idx(self, n_datas: int) -> None:
        """Advance the data window to the next batch of n_datas points.

        Args:
            n_datas (int): Number of data points to consume in this step.
        """
        self.idx_start = self.idx_end
        self.idx_end = self.idx_start + n_datas
        if self.idx_end == self.n_datas:
            self.all_data_consumed = True
        return None

    def _check_call_perform_one_step(self) -> None:
        """Raise an error if perform_one_step is called after all data has been consumed."""
        if self.all_data_consumed:
            raise SystemError(
                "{self} has been called whereas all his data have been consumed. You need to check all_data_consumed upstream."
            )
        return None

    def load_data(self, data: torch.Tensor, target: torch.Tensor) -> None:
        """Move training data to the worker's device and cache the dataset size.

        Args:
            data (torch.Tensor): Input features tensor.
            target (torch.Tensor): Labels tensor.
        """
        self.data = data.to(self.device)
        self.target = target.to(self.device)
        self.n_datas = len(self.target)
        return None

    def _init_variant(self, v: dict, n_layers: int) -> None:
        """Instantiate the variant configured in the guidelines.

        Args:
            v (dict): Variant sub-dict from the guidelines (keys: "name", "kwargs").
            n_layers (int): Number of trainable layers in the model.
        """
        self.is_variant = True
        self.variant_name: str = v["name"].__name__

        if self.variant_name == "ErrorFeedback":
            v["kwargs"]["n_layers"] = n_layers
            self.variant = v["name"](n_layers=n_layers, alpha=v["kwargs"]["alpha"])

        elif self.variant_name == "ProjFL":
            # ProjFL required the shape of the gradient
            layer_gradient_shapes = utils.get_shape_gradients(model=self.model)
            self.variant = v["name"](
                n_layers=n_layers,
                layer_gradient_shapes=layer_gradient_shapes,
                K=v["kwargs"]["K"],
                EF=v["kwargs"]["EF"],
            )

        elif self.variant_name == "ErrorFeedback21":
            self.variant = v["name"](n_layers=n_layers)

        elif self.variant_name == "Diana":
            self.variant = v["name"](n_layers=n_layers, alpha=v["kwargs"]["alpha"])

        else:
            raise NotImplementedError(f"unsupported variant: {self.variant_name}")

    def load_guidelines(self, guidelines: dict, n_layers: int) -> None:
        """Load training guidelines from the master and instantiate the variant if needed.

        Args:
            guidelines (dict): Master's parameter dictionary (optimizer, criterion,
                compressor, variant, etc.).
            n_layers (int): Number of trainable layers in the model.
        """
        self.guidelines = guidelines
        self.sparse_weight_updates = self.guidelines["sparse_weight_updates"]

        v = self.guidelines["variant"]
        if v["name"] is not None:
            self._init_variant(v=v, n_layers=n_layers)

        return None

    def reset_data_consumption(self) -> None:
        """Reset the data consumption state so the worker can process a new epoch."""
        self.all_data_consumed = False
        self.idx_start = 0
        self.idx_end = 0
        return None

    def _init_criterion(self) -> None:
        """Instantiate the loss criterion on the first step if not yet created."""
        if self.criterion is None:
            self.criterion = self.guidelines["criterion"]()
        return None

    def _init_optimizer(self) -> None:
        """Instantiate the optimizer on the first step if not yet created."""
        if self.optimizer is None:
            opt_class = self.guidelines["optimizer"]["name"]
            opt_kwargs = self.guidelines["optimizer"]["kwargs"]
            # create the instance
            self.optimizer = opt_class(self.model.parameters(), **opt_kwargs)
        return None

    def _init_compressor(self) -> None:
        """Instantiate the compressor on the first step if not yet created."""
        if self.compressor is None:
            cpr_class = self.guidelines["compressor"]["name"]
            if cpr_class is not None:
                self.is_compression = True
                self.compressor = cpr_class(**self.guidelines["compressor"]["kwargs"])
        return None

    def _find_relevant_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the current batch of inputs and labels using the stored index window.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input batch and corresponding labels.
        """
        inputs = self.data[self.idx_start : self.idx_end]
        labels = self.target[self.idx_start : self.idx_end]
        return inputs, labels

    def _forward_and_backward_pass(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Run the forward pass, compute the loss, and backpropagate gradients.

        Args:
            inputs (torch.Tensor): Input batch.
            labels (torch.Tensor): Corresponding labels.
        """

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # gradients are computed here
        loss.backward()
        return None

    def _task_update_weights_from_master(
        self, master_payload: OrderedDict[str, torch.Tensor]
    ) -> None:
        """Update the local model weights from the master's payload.

        Supports two modes:
        - Sparse update: only the indices modified by compressed gradients are updated.
        - Full update: the entire state dict is replaced (used on the first loading
          and when compression or sparse updates are disabled).

        Args:
            master_payload (OrderedDict[str, torch.Tensor]): Either a full state dict
                or a dict mapping layer names to (index, value) compressed tensors.
        """
        if (
            self.sparse_weight_updates
            and self.is_compression
            and not self.first_weight_loading
        ):
            current_parameters = self.model.state_dict()  # copy

            for layer_number, (layer_name, tensor) in enumerate(master_payload.items()):
                # Extract flattened indices and values
                idx_flat = tensor[:, 0].long()
                values = tensor[:, 1]

                # Convert flat indices to multi-dimensional indices
                multi_idx = torch.stack(
                    torch.unravel_index(
                        idx_flat, self.layer_gradient_shapes[layer_number]
                    )
                )

                # Update the current parameters in one shot
                current_parameters[layer_name].index_put_(tuple(multi_idx), values)

            # Load updated parameters into the model
            self.model.load_state_dict(current_parameters)

        else:
            self.model.load_state_dict(master_payload)

        if self.first_weight_loading:
            self.first_weight_loading = False

    def _compute_compressed_gradient(
        self,
        p: torch.nn.Parameter,
        layer_number: int,
        dic_results: dict[int, dict[str, torch.Tensor | float]],
    ) -> None:
        """Apply variant pre-processing, compress the gradient, and apply variant post-processing.

        Modifies dic_results in-place with the compressed gradient (and alpha for ProjFL).

        Args:
            p (torch.nn.Parameter): The model parameter whose gradient is being processed.
            layer_number (int): Index of the current layer.
            dic_results (dict): Result dict to update with the compressed gradient.
        """
        if self.is_variant:

            if self.variant_name == "ErrorFeedback":
                p.grad = self.variant.compute_gtild(
                    grad=p.grad, layer_number=layer_number
                )

            elif self.variant_name == "ProjFL":
                self.variant.update_dbar(layer_number=layer_number)
                alpha, p.grad = self.variant.compute_proj(
                    grad=p.grad, layer_number=layer_number
                )
                if self.variant.EF:
                    p.grad += self.variant.errors[layer_number]

                # store the results
                dic_results[layer_number]["alpha"] = alpha

            elif self.variant_name == "ErrorFeedback21":
                if self.variant.desc_dir[layer_number] is not None:
                    p.grad -= (
                        self.variant.gamma * self.variant.desc_dir[layer_number]
                    )

            elif self.variant_name == "Diana":
                if self.variant.h[layer_number] is not None:
                    p.grad -= self.variant.gamma * self.variant.h[layer_number]

            else:
                raise NotImplementedError(
                    f"unsupported variant: {self.variant_name}"
                )

        # apply the compressor
        compressed_gradient = self.compressor.compress(p)

        if self.is_variant:

            sparse_gradient = self.compressor.sparse(
                compressed_tensor=compressed_gradient, dense_shape=p.shape
            )

            if self.variant_name == "ErrorFeedback":
                self.variant.update_error(
                    gtild=p.grad,
                    gtild_sparse=sparse_gradient,
                    layer_number=layer_number,
                )

            elif self.variant_name == "ProjFL":
                # compressed_gradient = M_{t+1}^i
                self.variant.update_desc_dir(
                    sparse_grad_orth=sparse_gradient,
                    alpha=alpha,
                    layer_number=layer_number,
                )
                if self.variant.EF:
                    self.variant.update_error(
                        sparse_gradient=sparse_gradient,
                        gradorth_plus_error=p.grad,
                        layer_number=layer_number,
                    )

            elif self.variant_name == "ErrorFeedback21":
                self.variant.update_desc_dir(
                    sparse_grad_diff=sparse_gradient,
                    layer_number=layer_number,
                )

            elif self.variant_name == "Diana":
                self.variant.update_h(
                    m=sparse_gradient, layer_number=layer_number
                )

            else:
                raise NotImplementedError(
                    f"unsupported variant: {self.variant_name}"
                )

        dic_results[layer_number]["gradient"] = compressed_gradient

    def _task_perform_one_step(
        self, n_datas: int
    ) -> dict[int, dict[str, torch.Tensor | float]]:
        """Perform one gradient step and return the results for each layer.

        Initializes the criterion, optimizer, and compressor on the first call.
        Applies the configured variant (if any) before and after compression.

        Args:
            n_datas (int): Number of data points to use in this step.

        Returns:
            dict[int, dict[str, torch.Tensor | float]]: A dictionary mapping
                each layer index to a dict with keys "gradient", "alpha", and "err".
        """
        self._check_call_perform_one_step()

        # init procedure - create instance
        self._init_criterion()
        self._init_optimizer()
        self._init_compressor()

        self._update_idx(n_datas)
        inputs, labels = self._find_relevant_data()
        self._forward_and_backward_pass(inputs, labels)

        # will contain compressed gradients or dense gradient regarding the compressor parameter
        dic_results = {
            layer_number: {"gradient": None, "alpha": None, "err": None}
            for layer_number in range(len(self.layer_gradient_shapes))
        }

        for layer_number, p in enumerate(self.model.parameters()):

            if self.is_compression:
                self._compute_compressed_gradient(p, layer_number, dic_results)

            # no compression
            else:
                dic_results[layer_number]["gradient"] = p.grad

            # avoid memory leakage: https://github.com/pytorch/pytorch/issues/4661
            # UserWarning: Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient
            # which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this.
            # If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the
            # cycle and avoid the leak.
            # (Triggered internally at /opt/conda/conda-bld/pytorch_1711403380164/work/torch/csrc/autograd/engine.cpp:1177.)
            p.grad = None

        return dic_results
