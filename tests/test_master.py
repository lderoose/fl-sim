import copy

import pytest
import torch

from src.lib.compressor import TopKCompressor
from src.user.master import Master
from src.communication.communicator import LocalCommunicator
from src.utils import compute_n_layers, extract_trainable_name_layers
from tests.utils import Model42k4Layers


N_WORKERS = 2
DEVICE = "cpu"


def make_master(model, compressor_class=None, sparse_weight_updates=False):
    params = {
        "compressor": {"name": compressor_class},
        "variant": {"name": None},
        "sparse_weight_updates": sparse_weight_updates,
    }
    n_layers = compute_n_layers(model)
    communicator = LocalCommunicator(n_users=N_WORKERS + 1)
    m = Master(
        id="-1",
        communicator=communicator,
        device=DEVICE,
        params=params,
        n_workers=N_WORKERS,
        n_layers=n_layers,
    )
    m.load_model(copy.deepcopy(model))
    return m


@pytest.fixture
def model():
    return Model42k4Layers()


@pytest.fixture
def master_no_compression(model):
    return make_master(model, compressor_class=None)


@pytest.fixture
def master_with_compression(model):
    return make_master(model, compressor_class=TopKCompressor, sparse_weight_updates=True)


# _perform_aggregation
def test_perform_aggregation_equal_weights(master_no_compression) -> None:
    shapes = master_no_compression.layer_gradient_shapes
    gradients = {
        layer: [torch.ones(shape), torch.ones(shape) * 3.0]
        for layer, shape in shapes.items()
    }
    weights = torch.tensor([0.5, 0.5])
    aggregated = master_no_compression._perform_aggregation(gradients=gradients, weights=weights)

    for layer, shape in shapes.items():
        expected = torch.full(shape, 2.0)  # 0.5 * 1 + 0.5 * 3
        assert torch.allclose(aggregated[layer], expected)


def test_perform_aggregation_unequal_weights(master_no_compression) -> None:
    shapes = master_no_compression.layer_gradient_shapes
    gradients = {
        layer: [torch.zeros(shape), torch.ones(shape)]
        for layer, shape in shapes.items()
    }
    weights = torch.tensor([0.0, 1.0])
    aggregated = master_no_compression._perform_aggregation(gradients=gradients, weights=weights)

    for layer, shape in shapes.items():
        assert torch.allclose(aggregated[layer], torch.ones(shape))


def test_perform_aggregation_with_compression(master_with_compression) -> None:
    shapes = master_with_compression.layer_gradient_shapes
    # build compressed gradients: update only flat index 0 with value 1.0 for each worker
    compressed_gradients = {
        layer: [
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 1.0]]),
        ]
        for layer in shapes
    }
    weights = torch.tensor([0.5, 0.5])
    aggregated = master_with_compression._perform_aggregation(
        gradients=compressed_gradients, weights=weights
    )

    for layer, shape in shapes.items():
        result = aggregated[layer]
        assert result.shape == torch.Size(shape)
        flat = result.reshape(-1)
        assert torch.isclose(flat[0], torch.tensor(1.0))
        assert torch.equal(flat[1:], torch.zeros_like(flat[1:]))


# _extract_updated_index
def test_extract_updated_index_union(master_with_compression) -> None:
    n_layers = len(master_with_compression.layer_gradient_shapes)
    # worker 0 touches indices {0, 1}, worker 1 touches {1, 2}
    master_with_compression.worker_results["gradient"] = {
        layer: [
            torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
            torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        ]
        for layer in range(n_layers)
    }
    result = master_with_compression._extract_updated_index()
    for layer in range(n_layers):
        assert set(result[layer]) == {0, 1, 2}


def test_extract_updated_index_single_worker(master_with_compression) -> None:
    n_layers = len(master_with_compression.layer_gradient_shapes)
    master_with_compression.worker_results["gradient"] = {
        layer: [torch.tensor([[5.0, 9.0]])]
        for layer in range(n_layers)
    }
    master_with_compression.n_workers = 1
    result = master_with_compression._extract_updated_index()
    for layer in range(n_layers):
        assert set(result[layer]) == {5}


# build_payload
def test_build_payload_output_format(master_with_compression) -> None:
    n_layers = len(master_with_compression.layer_gradient_shapes)
    for layer in range(n_layers):
        master_with_compression.worker_results["gradient"][layer] = [
            torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
            torch.tensor([[0.0, 0.5], [2.0, 1.5]]),
        ]

    payload = master_with_compression.build_payload()
    layer_names = extract_trainable_name_layers(master_with_compression.model)

    assert set(payload.keys()) == set(layer_names)
    for tensor in payload.values():
        assert tensor.ndim == 2
        assert tensor.shape[1] == 2


def test_build_payload_indices_within_layer_size(master_with_compression) -> None:
    shapes = master_with_compression.layer_gradient_shapes
    n_layers = len(shapes)
    for layer in range(n_layers):
        master_with_compression.worker_results["gradient"][layer] = [
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 0.5]]),
        ]

    payload = master_with_compression.build_payload()

    for i, (_, tensor) in enumerate(payload.items()):
        n_elements = torch.tensor(shapes[i]).prod().item()
        indices = tensor[:, 0].long()
        assert (indices >= 0).all()
        assert (indices < n_elements).all()
