from abc import ABC, abstractmethod

import torch


class Compressor(ABC):
    """Abstract base class for gradient compressors.

    A compressor reduces the communication cost by selecting a subset of
    gradient coordinates and encoding them as (flat_index, value) pairs.
    Concrete subclasses must implement compress and compute_th_n_outputs.
    """

    @abstractmethod
    def compress(self) -> torch.Tensor:
        """Compress a gradient tensor.

        Returns:
            torch.Tensor: 2D tensor of shape (k, 2) where each row is
                [flat_index, value].
        """
        pass

    @abstractmethod
    def compute_th_n_outputs(self) -> int:
        """Return the theoretical number of scalar outputs produced by this compressor.

        Returns:
            int: Total number of scalar values in the compressed representation
                (i.e. 2 * k, since each element is stored as an index-value pair).
        """
        pass

    @staticmethod
    def sparse(
        compressed_tensor: torch.Tensor, dense_shape: tuple | list
    ) -> torch.Tensor:
        """Convert a compressed (index, value) tensor back to a sparse tensor.

        Args:
            compressed_tensor (torch.Tensor): 2D tensor of shape (k, 2) with
                columns [flat_index, value].
            dense_shape (tuple | list): Shape of the original dense tensor.

        Returns:
            torch.Tensor: Sparse tensor of shape dense_shape with zero values
                at unselected positions.
        """
        flat_indices = compressed_tensor[:, 0].long()
        values = compressed_tensor[:, 1]
        multi_indices = torch.stack(torch.unravel_index(flat_indices, dense_shape))
        sparse_tensor = torch.zeros(
            dense_shape, dtype=values.dtype, device=values.device
        )
        sparse_tensor.index_put_(tuple(multi_indices), values)
        return sparse_tensor


class KCompressor(Compressor):
    """Base class for compressors that select exactly k gradient coordinates.

    The number of selected coordinates k can be specified as an absolute integer
    or as a factor of the total number of elements.
    """

    def __init__(self, k: int | float, k_as_factor: bool) -> None:
        """Initialize the k-based compressor.

        Args:
            k (int | float): Number of coordinates to keep. If k_as_factor
                is True, this is treated as a fraction of the total elements and
                must belong to (0, 0.5). If k_as_factor is False, this is an
                absolute count and must be a positive integer.
            k_as_factor (bool): Whether k is interpreted as a fraction.
        """
        super().__init__()
        self.k = k
        self.k_as_factor = k_as_factor

        self._check_k(k, k_as_factor)
        return None

    def _check_k(self, k: int | float, k_as_factor: bool) -> None:
        """Validate the k and k_as_factor parameters.

        When k_as_factor is True, k must be a float in (0, 0.5) to guarantee
        a reduction in communication cost. Each compressed element is stored as
        a (index, value) pair, so k > 0.5 would result in more data being sent
        than the original tensor.

        Args:
            k (int | float): Value of k to validate.
            k_as_factor (bool): Whether k is a fraction.

        Raises:
            TypeError: If k has the wrong type.
            ValueError: If k is out of the valid range.
        """
        if k_as_factor:
            err_msg = f"parameter 'k_as_factor' enable, 'k' must be a float belonging to ]0, 0.5[ to reduce the communication cost"
            if not isinstance(k, float):
                raise TypeError(err_msg)
            else:
                if (k <= 0) or (k >= 0.5):
                    raise ValueError(err_msg)
        else:
            err_msg = f"parameter 'k_as_factor' disable, 'k' must be a positive integer"
            if not isinstance(k, int):
                raise TypeError(err_msg)
            else:
                if k < 1:
                    raise ValueError(err_msg)
        return None

    def compute_th_n_outputs(self, input_tensor: torch.Tensor) -> int:
        """Compute the theoretical number of scalar outputs for a given input tensor.

        Args:
            input_tensor (torch.Tensor): The tensor to be compressed.

        Returns:
            int: Number of scalar values in the compressed representation.
        """
        if not self.k_as_factor:
            # if one element in the input_tensor and k>1
            return 2 * min(self.k, input_tensor.numel())
        else:
            return 2 * max(1, int(input_tensor.numel() * self.k))


class TopKCompressor(KCompressor):
    """Compressor that retains the k gradient coordinates with the largest absolute values."""

    def compress(self, weights: torch.Tensor) -> torch.Tensor:
        """Compress a gradient tensor by keeping the k largest-magnitude entries.

        The output is a 2D tensor where each row is [flat_index, value]:
        - flat_index: position of the element in the flattened gradient.
        - value: the gradient value at that position.

        The number of rows depends on k, k_as_factor, and the input tensor size.

        Args:
            weights (torch.Tensor): Parameter tensor whose .grad attribute holds
                the gradient to compress.

        Returns:
            torch.Tensor: Compressed tensor of shape (k_actual, 2).
        """
        # update k
        n_outputs = self.compute_th_n_outputs(input_tensor=weights)
        k_up = n_outputs // 2

        flat = weights.grad.reshape(-1)  # to 1D (flatten)
        _, idx = flat.abs().topk(k_up, largest=True, sorted=True)
        return torch.stack((idx, flat[idx]), dim=1)


class RandomKCompressor(KCompressor):
    """Unbiased compressor that retains k randomly selected gradient coordinates."""

    def compress(self, weights: torch.Tensor) -> torch.Tensor:
        """Compress a gradient tensor by randomly sampling k entries (unbiased).

        Selected values are scaled by n/k to make the estimator unbiased. The
        output is a 2D tensor where each row is [flat_index, scaled_value].

        The number of rows depends on k, k_as_factor, and the input tensor size.

        Args:
            weights (torch.Tensor): Parameter tensor whose .grad attribute holds
                the gradient to compress.

        Returns:
            torch.Tensor: Compressed tensor of shape (k_actual, 2).
        """
        # update k
        n_outputs = self.compute_th_n_outputs(input_tensor=weights)
        k_up = n_outputs // 2

        # make it unbiased
        c = weights.numel() / k_up if self.k_as_factor else weights.numel() / self.k

        flat = weights.grad.reshape(-1)  # to 1D (flatten)
        idx = torch.randperm(flat.numel(), device=flat.device)[:k_up]
        return torch.stack((idx, c * flat[idx]), dim=1)
