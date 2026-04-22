import pytest

import torch

from src.user.master import Master
from src.user.worker import Worker
from conf.convention import MASTER_ID
from src.communication.communicator import LocalCommunicator


# settings
fake_params = {
    "compressor": {"name": None},
    "variant": {"name": None},
    "sparse_weight_updates": True,
}

N_LAYERS = 5
N_WORKERS = 3
DEVICE = "cpu"
MSG_DATA_SEND = "HELLO WORLD"

communicator = LocalCommunicator(n_users=N_WORKERS + 1, track_cc=True)
master = Master(
    id="-1",
    communicator=communicator,
    device="cpu",
    params=fake_params,
    n_workers=N_WORKERS,
    n_layers=N_LAYERS,
)

ls_workers: list[Worker] = []
for i in range(N_WORKERS):
    w = Worker(
        id=f"{i}",
        communicator=communicator,
        device=DEVICE,
    )
    ls_workers.append(w)

ls_workers_id = [w.id for w in ls_workers]


def test_master_send_worker_receive() -> None:
    communicator.reset_queues()
    master.send_message(to=ls_workers_id, subject="test", data=MSG_DATA_SEND)
    for worker in ls_workers:
        msg_received = worker.receive_message()
        assert (
            msg_received.data == MSG_DATA_SEND
        ), f"expected '{MSG_DATA_SEND}', get '{msg_received.data}' instead"
    return None


def test_worker_send_master_receive() -> None:
    communicator.reset_queues()
    for worker in ls_workers:
        worker.send_message(to=MASTER_ID, subject="test", data=MSG_DATA_SEND)

    for worker_id in ls_workers_id:
        msg_received = master.receive_message(from_=worker_id)
        assert (
            msg_received.data == MSG_DATA_SEND
        ), f"expected '{MSG_DATA_SEND}', get '{msg_received.data}' instead"
    return None


def test_many_message_in_one_queue() -> None:
    communicator.reset_queues()
    master.send_message(to=ls_workers_id[0], subject="test", data=MSG_DATA_SEND)
    ls_workers[0].send_message(to=MASTER_ID, subject="test", data=MSG_DATA_SEND)

    # sending an additional message should trigger an error
    with pytest.raises(ValueError):
        master.send_message(to=ls_workers_id[0], subject="test", data=MSG_DATA_SEND)

    with pytest.raises(ValueError):
        ls_workers[0].send_message(to=MASTER_ID, subject="test", data=MSG_DATA_SEND)
    return None
