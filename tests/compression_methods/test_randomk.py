import torch
import pytest
import numpy as np


from tests.utils import FakeTensor
from src.lib.compressor import RandomKCompressor


@pytest.mark.parametrize(
    "k, k_as_factor, expected_exception",
    [
        (-1, False, ValueError),
        (0.5, False, TypeError),
        (0, False, ValueError),
        (-1, True, TypeError),
        (0.7, True, ValueError),
    ],
)
def test_randomk_init(k, k_as_factor, expected_exception) -> None:
    with pytest.raises(expected_exception):
        RandomKCompressor(k, k_as_factor)
    return None


@pytest.mark.parametrize(
    "k, k_as_factor",
    [
        (3, False),
        (10, False),
        (1_000_000_000, False),
        (0.25, True),
        (0.01, True),
        (0.0001, True),
    ],
)
def test_randomk(k, k_as_factor, tensors) -> None:
    cpr = RandomKCompressor(k=k, k_as_factor=k_as_factor)  # init

    for dense_tensor in tensors:
        fake_dense_tensor = FakeTensor(dense_tensor)
        cpr_tensor = cpr.compress(weights=fake_dense_tensor)

        # compute k
        n_components = dense_tensor.numel()
        if not k_as_factor:
            # if one element in the input_tensor and k>1
            k_up = min(k, n_components)
        else:
            k_up = max(int(n_components * k), 1)

        # check the shape
        expected_shape = [k_up, 2]
        assert list(cpr_tensor.shape) == expected_shape
    return None


@pytest.mark.parametrize(
    "k, k_as_factor",
    [
        (3, False),
        (10, False),
        (1_000_000_000, False),
        (0.25, True),
        (0.01, True),
        (0.0001, True),
    ],
)
def test_randomk_compressor_n_outputs(
    k, k_as_factor, tensors: list[torch.Tensor]
) -> None:
    cpr = RandomKCompressor(k=k, k_as_factor=k_as_factor)  # init

    for dense_tensor in tensors:
        # convert to FakeTensor to have the .grad attribute
        dense_tensor = FakeTensor(dense_tensor)

        n_outputs_th = cpr.compute_th_n_outputs(dense_tensor)
        compressed_tensor = cpr.compress(dense_tensor)
        n_outputs_emp = compressed_tensor.numel()

    err_msg = f"expected the theoretical number of outputs '{n_outputs_th}' equals to \
        the empirical number of outputs '{n_outputs_emp}'"
    assert n_outputs_th == n_outputs_emp, err_msg
    return None
