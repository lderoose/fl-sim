from typing import Callable

import torch


class GradientAggregate:
    """Utility class for aggregating per-worker gradients into a single gradient per layer.

    All methods are static. Supports aggregation of both dense/sparse and
    compressed gradient representations.
    """

    @staticmethod
    def from_sparse(
        dic_gradients: dict, device: str, weights: torch.Tensor = None
    ) -> list:
        """Aggregate per-worker gradients using a weighted mean.

        Args:
            dic_gradients (dict): Mapping from layer index to a list of dense or sparse
                gradient tensors, one per worker.
            device (str): Compute device string (e.g. "cpu" or "cuda:x").
            weights (torch.Tensor, optional): 1D tensor of per-worker weights. If None,
                each worker receives equal weight (1 / n_workers).

        Returns:
            list: Aggregated gradient tensor for each layer.
        """
        # need to check if all workers are involved in this step
        n_workers = len(dic_gradients[0])

        if weights is None:
            # consider each worker has the same weight
            fv = 1 / n_workers
            weights = torch.full(size=(n_workers,), fill_value=fv)
        else:
            weights = weights[
                :n_workers
            ]  # in case some workers are not involved they have 0 value as weight

        ls_up_gradients = []

        for ls_gradients_layer in dic_gradients.values():

            tensor_gradient_layer = torch.stack(ls_gradients_layer)
            dim_gradient = ls_gradients_layer[0].dim()

            # with this shape, the weight is apply to the gradient of the worker
            dim_reshape = tuple([-1] + [1] * (dim_gradient))

            # multiply all the values of the gradient for this worker, for this layer by the weight for this worker
            tensor_weighted_gradient_layer = torch.mul(
                tensor_gradient_layer.to(device), weights.view(*dim_reshape).to(device)
            )

            # because the sum of weights=1 (see compute_weights), just sum up
            g_bar_layer = tensor_weighted_gradient_layer.sum(0)

            # save it
            ls_up_gradients.append(g_bar_layer)

        return ls_up_gradients

    @staticmethod
    def compressed_to_sparse(
        dic_compressed_gradients: dict, layer_gradient_shapes: dict, f_cts: Callable
    ) -> dict:
        """Reconstruct sparse gradient tensors from compressed representations for each layer.

        Args:
            dic_compressed_gradients (dict): Mapping from layer index to a list of compressed
                gradient objects.
            layer_gradient_shapes (dict): Mapping from layer index to the original dense shape
                of that layer's gradient.
            f_cts (Callable): Function that converts a compressed tensor back to a sparse tensor.
                Signature: f_cts(compressed_tensor, dense_shape) -> torch.Tensor.

        Returns:
            dict: Mapping from layer index to a list of reconstructed sparse gradient tensors.
        """
        dic_sparse_gradients = {layer: [] for layer in dic_compressed_gradients.keys()}

        for layer, ls_compressed_gradients in dic_compressed_gradients.items():
            original_shape = layer_gradient_shapes[layer]
            for compressed_gradient in ls_compressed_gradients:
                sparse_gradient_rebuild = f_cts(
                    compressed_tensor=compressed_gradient, dense_shape=original_shape
                )
                dic_sparse_gradients[layer].append(sparse_gradient_rebuild)
        return dic_sparse_gradients

    @staticmethod
    def from_compressed(
        dic_compressed_gradients: dict,
        layer_gradient_shapes: dict,
        f_cts: Callable,
        device: str,
        weights: torch.Tensor = None,
    ) -> list:
        """Aggregate compressed per-worker gradients by first decompressing then averaging.

        Decompresses each worker's compressed gradient to a sparse tensor using f_cts,
        then delegates to from_sparse for the weighted aggregation.

        Args:
            dic_compressed_gradients (dict): Mapping from layer index to a list of
                compressed (index, value) tensors, one per worker.
            layer_gradient_shapes (dict): Mapping from layer index to the original
                dense gradient shape.
            f_cts (Callable): Function that converts a compressed tensor to a sparse
                tensor. Signature: f_cts(compressed_tensor, dense_shape) -> Tensor.
            device (str): Compute device string.
            weights (torch.Tensor, optional): Per-worker weight tensor. If None,
                equal weights are used.

        Returns:
            list: Aggregated gradient tensor for each layer.
        """
        dic_sparse_gradients = GradientAggregate.compressed_to_sparse(
            dic_compressed_gradients=dic_compressed_gradients,
            layer_gradient_shapes=layer_gradient_shapes,
            f_cts=f_cts,
        )

        # now we have a dic with this schema {layer_i: [g0, g1, ..., gk], ...}, so just apply the from_sparse method
        return GradientAggregate.from_sparse(
            dic_gradients=dic_sparse_gradients, device=device, weights=weights
        )
