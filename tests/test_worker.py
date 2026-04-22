import copy

import pytest
import torch

from src.lib.compressor import TopKCompressor
from src.lib.variant import Diana, ErrorFeedback, ErrorFeedback21, ProjFL
from src.user.worker import Worker
from src.communication.communicator import LocalCommunicator
from src.utils import compute_n_layers, extract_trainable_name_layers
from tests.utils import Model42k4Layers


DEVICE = "cpu"


def make_guidelines(
    compressor_class=None,
    variant_class=None,
    variant_kwargs=None,
    sparse_weight_updates=False,
):
    if variant_kwargs is None:
        variant_kwargs = {"alpha": 0.5, "beta": 0.9, "K": 1, "EF": False}
    return {
        "compressor": {"name": compressor_class, "kwargs": {"k": 0.1, "k_as_factor": True}},
        "variant": {"name": variant_class, "kwargs": variant_kwargs},
        "sparse_weight_updates": sparse_weight_updates,
        "optimizer": {"name": torch.optim.SGD, "kwargs": {"lr": 0.1}},
        "criterion": torch.nn.MSELoss,
    }


@pytest.fixture
def model():
    return Model42k4Layers()


@pytest.fixture
def worker(model):
    communicator = LocalCommunicator(n_users=2)
    w = Worker(id="0", communicator=communicator, device=DEVICE)
    w.load_model(copy.deepcopy(model))
    return w


# load_guidelines
def test_load_guidelines_no_variant_no_compressor(worker, model) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(guidelines=make_guidelines(), n_layers=n_layers)

    assert worker.sparse_weight_updates is False
    assert worker.is_variant is False
    assert worker.variant is None


def test_load_guidelines_sparse_weight_updates_true(worker, model) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(
        guidelines=make_guidelines(
            compressor_class=TopKCompressor, sparse_weight_updates=True
        ),
        n_layers=n_layers,
    )
    assert worker.sparse_weight_updates is True


@pytest.mark.parametrize(
    "variant_class, variant_kwargs",
    [
        (ErrorFeedback, {"alpha": 0.5}),
        (ErrorFeedback21, {}),
        (Diana, {"alpha": 0.5, "beta": 0.9}),
    ],
)
def test_load_guidelines_variant_initialized(worker, model, variant_class, variant_kwargs) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(
        guidelines=make_guidelines(
            compressor_class=TopKCompressor,
            variant_class=variant_class,
            variant_kwargs={**{"alpha": 0.5, "beta": 0.9, "K": 1, "EF": False}, **variant_kwargs},
        ),
        n_layers=n_layers,
    )
    assert worker.is_variant is True
    assert worker.variant_name == variant_class.__name__
    assert worker.variant is not None


def test_load_guidelines_projfl_initialized(worker, model) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(
        guidelines=make_guidelines(
            compressor_class=TopKCompressor,
            variant_class=ProjFL,
            variant_kwargs={"K": 2, "EF": False},
        ),
        n_layers=n_layers,
    )
    assert worker.is_variant is True
    assert worker.variant_name == "ProjFL"
    assert worker.variant is not None


def test_load_guidelines_unknown_variant_raises(worker, model) -> None:
    class UnknownVariant:
        __name__ = "UnknownVariant"

    n_layers = compute_n_layers(model)
    with pytest.raises(NotImplementedError):
        worker.load_guidelines(
            guidelines=make_guidelines(
                compressor_class=TopKCompressor,
                variant_class=UnknownVariant,
            ),
            n_layers=n_layers,
        )


# _task_update_weights_from_master
def test_update_weights_full_state_dict(worker, model) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(guidelines=make_guidelines(), n_layers=n_layers)

    new_model = Model42k4Layers()
    new_state_dict = new_model.state_dict()

    worker._task_update_weights_from_master(master_payload=new_state_dict)

    for key in new_state_dict:
        assert torch.equal(worker.model.state_dict()[key], new_state_dict[key])


def test_update_weights_full_state_dict_sets_first_loading_false(worker, model) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(guidelines=make_guidelines(), n_layers=n_layers)
    assert worker.first_weight_loading is True

    worker._task_update_weights_from_master(master_payload=worker.model.state_dict())
    assert worker.first_weight_loading is False


def test_update_weights_sparse_updates_index(worker, model) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(
        guidelines=make_guidelines(compressor_class=TopKCompressor, sparse_weight_updates=True),
        n_layers=n_layers,
    )
    worker.first_weight_loading = False
    worker._init_compressor()

    layer_names = extract_trainable_name_layers(worker.model)
    first_layer = layer_names[0]

    # record original flat value at index 0
    original_val = worker.model.state_dict()[first_layer].reshape(-1)[0].clone()
    new_val = original_val + 100.0

    # sparse payload: update only flat index 0 of the first layer
    sparse_payload = {first_layer: torch.tensor([[0.0, new_val.item()]])}
    worker._task_update_weights_from_master(master_payload=sparse_payload)

    updated_val = worker.model.state_dict()[first_layer].reshape(-1)[0]
    assert torch.isclose(updated_val, new_val)


def test_update_weights_sparse_does_not_touch_other_indices(worker, model) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(
        guidelines=make_guidelines(compressor_class=TopKCompressor, sparse_weight_updates=True),
        n_layers=n_layers,
    )
    worker.first_weight_loading = False
    worker._init_compressor()

    layer_names = extract_trainable_name_layers(worker.model)
    first_layer = layer_names[0]

    original_flat = worker.model.state_dict()[first_layer].reshape(-1).clone()

    # only update flat index 0
    sparse_payload = {first_layer: torch.tensor([[0.0, 999.0]])}
    worker._task_update_weights_from_master(master_payload=sparse_payload)

    updated_flat = worker.model.state_dict()[first_layer].reshape(-1)
    # all indices except 0 must remain unchanged
    assert torch.equal(updated_flat[1:], original_flat[1:])


def test_update_weights_first_loading_uses_full_state_dict(worker, model) -> None:
    n_layers = compute_n_layers(model)
    worker.load_guidelines(
        guidelines=make_guidelines(compressor_class=TopKCompressor, sparse_weight_updates=True),
        n_layers=n_layers,
    )
    assert worker.first_weight_loading is True
    worker._init_compressor()

    new_model = Model42k4Layers()
    new_state_dict = new_model.state_dict()

    # even with sparse_weight_updates=True, first call must use full state dict
    worker._task_update_weights_from_master(master_payload=new_state_dict)

    for key in new_state_dict:
        assert torch.equal(worker.model.state_dict()[key], new_state_dict[key])
