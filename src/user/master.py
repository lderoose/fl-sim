from typing import Optional

import torch
from torch import nn

from conf.convention import MASTER_ID
from src import utils
from src.communication.communicator import Communicator
from src.communication.message import Message
from src.lib.gradient_aggregate import GradientAggregate
from src.lib.stopping_criterion import EarlyStopping
from src.user.user import User


class Master(User):
    """The central coordinator in the federated learning simulation.

    The Master aggregates gradients received from workers, updates the global
    model, and broadcasts the updated weights back. It also manages the optimizer,
    learning rate scheduler, and optional stopping criterion.
    """

    def __init__(
        self,
        id: str,
        communicator: Communicator,
        device: str,
        params: dict,
        n_workers: int,
        n_layers: int,
        log_level: Optional[str] = None,
    ) -> None:
        """Initialize the master.

        Args:
            id (str): Unique identifier (always MASTER_ID).
            communicator (Communicator): Shared communicator instance.
            device (str): Compute device string (e.g. "cpu" or "cuda:x").
            params (dict): Master parameter dictionary (optimizer, criterion,
                compressor, variant, etc.).
            n_workers (int): Total number of workers in the simulation.
            n_layers (int): Number of trainable layers in the model.
            log_level (Optional[str]): Python logging level. If None, logging is disabled.
        """
        super().__init__(MASTER_ID, communicator, device, log_level)  # id is always -1
        self.params = params
        self.n_layers = n_layers
        self.n_workers = n_workers

        self.lr_scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
        self.variants: list = []
        self.compressor = self.params["compressor"]
        self.is_compression = True if self.compressor["name"] is not None else False
        self.use_sparse_weight_updates = self.params["sparse_weight_updates"]

        self.variant_name: str = (
            self.params["variant"]["name"].__name__
            if self.params["variant"]["name"] is not None
            else ""
        )
        self.is_variant = True if self.params["variant"]["name"] is not None else False

        self.stopping_criterion: Optional[EarlyStopping] = None

        self.worker_results = self.reset_worker_results()
        return None

    def reset_worker_results(
        self,
    ) -> dict[str, dict[int, list[torch.Tensor | float | BaseException]]]:
        """Reset the storage for worker results at the start of each step.

        Returns:
            dict: Empty result containers for "gradient", "alpha", and "err" per layer.
        """
        return {
            "gradient": {
                layer_number: [None for _ in range(self.n_workers)]
                for layer_number in range(self.n_layers)
            },
            "alpha": {
                layer_number: [None for _ in range(self.n_workers)]
                for layer_number in range(self.n_layers)
            },
            "err": {
                layer_number: [None for _ in range(self.n_workers)]
                for layer_number in range(self.n_layers)
            },
        }

    def load_model(self, model: nn.Module) -> None:
        """Load the model and set it to evaluation mode.

        Overrides User.load_model: the master never trains the model, it only evaluates it.

        Args:
            model (nn.Module): PyTorch model to load.
        """
        super().load_model(model)
        self.model.eval()
        return None

    def receive_message(self, from_: str) -> Optional[Message]:
        """Retrieve and dispatch the next message from a worker.

        Supported subjects:
            - "one_step_result": store the gradient results sent by the worker.
            - "test": used in pytest only.

        Args:
            from_ (str): Identifier of the sender worker.

        Returns:
            Optional[Message]: The message for "test" subjects, None otherwise.
        """
        msg: Message = super().receive_message(from_)

        # pytest
        if msg.subject == "test":
            return msg

        elif msg.subject == "one_step_result":
            self._task_receive_results_from_worker(msg)

        else:
            raise NotImplementedError(f"{msg.subject} feature not implemented")

        return None

    def _store_worker_results(self, msg: Message) -> None:
        """Store gradient and alpha results from a worker message.

        Args:
            msg (Message): Message from a worker with subject "one_step_result".
        """
        for layer_number, results in msg.data.items():
            self.worker_results["gradient"][layer_number][int(msg.src)] = results[
                "gradient"
            ]

            if results["alpha"] is not None:
                self.worker_results["alpha"][layer_number][int(msg.src)] = results[
                    "alpha"
                ]
        return None

    def _task_receive_results_from_worker(self, msg: Message) -> None:
        """Handle a one-step result message from a worker.

        Args:
            msg (Message): Message from a worker with subject "one_step_result".
        """
        self._store_worker_results(msg)
        return None

    def _init_training(self) -> None:
        """Initialize the optimizer, lr_scheduler, and per-worker variant instances.

        Called on the first invocation of update_model. Sets up one variant instance
        per worker for algorithms that maintain worker-level state (ProjFL, EF21,
        EF21Plus, Diana), and creates the lr_scheduler if configured.
        """
        if self.variant_name == "ProjFL":

            layer_gradient_shapes = utils.get_shape_gradients(model=self.model)
            var = self.params["variant"]

            for _ in range(self.n_workers):
                v = var["name"](
                    n_layers=self.n_layers,
                    layer_gradient_shapes=layer_gradient_shapes,
                    K=var["kwargs"]["K"],
                    EF=var["kwargs"]["EF"],
                )
                self.variants.append(v)

        elif self.variant_name == "ErrorFeedback21":
            for _ in range(self.n_workers):
                v = self.params["variant"]["name"](n_layers=self.n_layers)
                self.variants.append(v)

        elif self.variant_name == "Diana":
            var = self.params["variant"]
            for _ in range(self.n_workers):
                v = self.params["variant"]["name"](
                    n_layers=self.n_layers,
                    alpha=var["kwargs"]["alpha"],
                    beta=var["kwargs"]["beta"],
                )
                self.variants.append(v)

        if self.params["lr_scheduler"]["name"] is not None:
            self.lr_scheduler = self.params["lr_scheduler"]["name"](
                optimizer=self.optimizer, **self.params["lr_scheduler"]["kwargs"]
            )
        return None

    def _init_stopping_criterion(self) -> None:
        """Instantiate the stopping criterion on the first call if not yet created."""
        if self.stopping_criterion is None:
            sc_class = self.params["stopping_criterion"]["name"]
            if sc_class is not None:
                sc_kwargs = self.params["stopping_criterion"]["kwargs"]
                if self.logger is not None:
                    sc_kwargs["trace_func"] = self.logger.debug
                else:
                    sc_kwargs["trace_func"] = print
                self.stopping_criterion = sc_class(**sc_kwargs)
        return None

    def _perform_aggregation(self, gradients: dict, weights: dict) -> list:
        """Aggregate per-worker gradients into a single gradient per layer.

        Uses dense aggregation when no compressor is configured, and sparse
        (decompressed) aggregation when a compressor is active.

        Args:
            gradients (dict): Per-layer list of worker gradients.
            weights (dict): Per-step weight tensor used for the weighted mean.

        Returns:
            list: Aggregated gradient tensor for each layer.
        """
        if self.compressor["name"] is None:
            ls_updated_gradients = GradientAggregate.from_sparse(
                dic_gradients=gradients, device=self.device, weights=weights
            )

        else:
            ls_updated_gradients = GradientAggregate.from_compressed(
                dic_compressed_gradients=gradients,
                layer_gradient_shapes=self.layer_gradient_shapes,
                f_cts=self.compressor["name"].sparse,
                device=self.device,
                weights=weights,
            )
        return ls_updated_gradients

    def _aggregate_gradients(self, weights) -> list:
        """Compute the aggregated gradient per layer, applying variant logic if configured.

        Handles all supported variants (ProjFL, ErrorFeedback21, Diana, ErrorFeedback)
        as well as the no-variant case.

        Args:
            weights: Per-step weight tensor for the weighted gradient aggregation.

        Returns:
            list: Aggregated gradient tensor for each layer.
        """
        if self.is_variant:

            # put the gradients in the correct shape
            dic_sparse_gradients = GradientAggregate.compressed_to_sparse(
                dic_compressed_gradients=self.worker_results["gradient"],
                layer_gradient_shapes=self.layer_gradient_shapes,
                f_cts=self.compressor["name"].sparse,
            )

            if self.variant_name == "ProjFL":
                dic_dt = {i: [] for i in range(len(self.worker_results["gradient"]))}

                for layer_number, gradients in dic_sparse_gradients.items():
                    for worker_number, gradient in enumerate(gradients):
                        self.variants[worker_number].update_dbar(
                            layer_number=layer_number
                        )

                        self.variants[worker_number].update_desc_dir(
                            sparse_grad_orth=gradient,
                            alpha=self.worker_results["alpha"][layer_number][
                                worker_number
                            ],
                            layer_number=layer_number,
                        )

                        dic_dt[layer_number].append(
                            self.variants[worker_number].desc_dir[layer_number][-1]
                        )

                return GradientAggregate.from_sparse(
                    dic_gradients=dic_dt, device=self.device, weights=weights
                )

            elif self.variant_name == "ErrorFeedback21":
                dic_dt = {i: [] for i in range(len(self.worker_results["gradient"]))}

                for layer_number, gradients in dic_sparse_gradients.items():
                    for worker_number, gradient in enumerate(gradients):
                        self.variants[worker_number].update_desc_dir(
                            sparse_grad_diff=gradient,
                            layer_number=layer_number,
                        )

                        dic_dt[layer_number].append(
                            self.variants[worker_number].desc_dir[layer_number]
                        )
                return GradientAggregate.from_sparse(
                    dic_gradients=dic_dt, device=self.device, weights=weights
                )

            elif self.variant_name == "Diana":
                dic_dt = {i: [] for i in range(len(self.worker_results["gradient"]))}

                for layer_number, gradients in dic_sparse_gradients.items():
                    for worker_number, gradient in enumerate(gradients):
                        self.variants[worker_number].update_desc_dir(
                            sparse_grad=gradient,
                            layer_number=layer_number,
                        )

                        self.variants[worker_number].update_h(
                            m=gradient, layer_number=layer_number
                        )

                        dic_dt[layer_number].append(
                            self.variants[worker_number].desc_dir[layer_number]
                        )
                return GradientAggregate.from_sparse(
                    dic_gradients=dic_dt, device=self.device, weights=weights
                )

            elif self.variant_name == "ErrorFeedback":
                return self._perform_aggregation(
                    gradients=self.worker_results["gradient"], weights=weights
                )

            else:
                raise ValueError(
                    f"define a rule at the master level for this 'variant': {self.variant_name}"
                )

        # case without variant
        else:
            return self._perform_aggregation(
                gradients=self.worker_results["gradient"], weights=weights
            )

    def update_model(
        self,
        weights=None,
    ) -> None:
        """Aggregate worker gradients and perform one optimizer step on the global model.

        Initializes the optimizer and variant instances on the first call.

        Args:
            weights: Per-step weight tensor for the weighted gradient aggregation.
        """
        if self.optimizer is None:
            opt_class = self.params["optimizer"]["name"]
            opt_kwargs = self.params["optimizer"]["kwargs"]
            self.optimizer = opt_class(self.model.parameters(), **opt_kwargs)
            self._init_training()

        ls_updated_gradients = self._aggregate_gradients(weights=weights)

        # update master's gradients
        for i, p in enumerate(self.model.parameters()):
            p.grad = ls_updated_gradients[i]

        # update model weights, the weight decay is applied here only
        self.optimizer.step()

        self.reset_worker_results()

        return None

    def build_payload(self) -> dict[str, torch.Tensor]:
        """Build a compressed weight payload mapping each trainable layer to its updated values.

        For each layer, collects the flat indices that were modified by at least one
        worker and packs them together with the corresponding current parameter values
        into a (N, 2) tensor of the form [[index, value], ...].

        Returns:
            dict[str, torch.Tensor]: Mapping from layer name to (N, 2) update tensor.
        """
        dic_updated_index = self._extract_updated_index()

        # Extract all trainable layer keys
        ls_keys = utils.extract_trainable_name_layers(self.model)
        dic_out = {k: None for k in ls_keys}

        # Mapping between layer numbers and model weight keys
        mapping_keys = {
            key_a: key_b for key_a, key_b in zip(dic_updated_index.keys(), ls_keys)
        }

        for layer_number, ls_updated_idx in dic_updated_index.items():
            key_in_mw = mapping_keys[layer_number]

            # Convert to tensor once, keep on same device as model
            idx_tensor = torch.tensor(
                ls_updated_idx,
                device=self.device,
                dtype=torch.long,
            )

            # Flatten layer weights once
            layer_tensor = self.model.state_dict()[key_in_mw].reshape(-1)

            # Vectorized value extraction
            selected_values = layer_tensor[idx_tensor]

            # Stack indices and values into a (N, 2) tensor
            output = torch.stack(
                [idx_tensor.to(selected_values.dtype), selected_values], dim=1
            )

            dic_out[key_in_mw] = output

        return dic_out

    def _extract_updated_index(self) -> dict[str, list[torch.Tensor]]:
        """Collect the unique flat indices modified by at least one worker across all layers.

        Returns:
            dict[int, list]: Mapping from layer number to a list of unique flat indices.
        """
        dic_idx_up = {}
        gradients = self.worker_results["gradient"]

        for layer_number, ls_tensors in gradients.items():
            all_indices = torch.cat([t[:, 0].int() for t in ls_tensors])
            unique_indices = torch.unique(all_indices)
            dic_idx_up[layer_number] = unique_indices.tolist()

        return dic_idx_up

    def update_scheduler(self, loss: float) -> float:
        """Step the learning rate scheduler and return the new learning rate.

        Args:
            loss (float): Validation loss used to drive the scheduler.

        Returns:
            float: Updated learning rate.
        """
        self.current_lr = self.optimizer.param_groups[0]["lr"]
        self.lr_scheduler.step(loss)
        lr = self.optimizer.param_groups[0]["lr"]
        if lr != self.current_lr:
            print(f"LR has been updated from {self.current_lr} to {lr}")
        return lr

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute the loss on the given inputs without updating model parameters.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Ground-truth labels.

        Returns:
            float: Scalar loss value.
        """
        if not self.model.training:
            with torch.no_grad():
                pred = self.model(x)
        else:
            raise ValueError(f"the master model shouldn't be in training mode.")
        f_loss = self.params["criterion"]()
        loss = f_loss(pred, y)
        return loss.item()

    def monitor_stopping(self, x_val: torch.Tensor, y_val: torch.Tensor) -> bool:
        """Evaluate the stopping criterion on the validation set.

        Args:
            x_val (torch.Tensor): Validation input data.
            y_val (torch.Tensor): Validation labels.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        self._init_stopping_criterion()
        loss_value = self.compute_loss(x_val, y_val)

        self.stopping_criterion(epoch_score=loss_value, model=self.model)
        if self.stopping_criterion.early_stop:
            return True
        else:
            return False
