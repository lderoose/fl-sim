from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Optional

from conf.convention import MASTER_ID
from src.communication.communication_cost import CommunicationCostTracker
from src.communication.message import Message


class Communicator(ABC):
    """Abstract base class for inter-user communication channels."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the communicator."""

    @abstractmethod
    def send(self) -> None:
        """Send a message from one user to another."""

    @abstractmethod
    def receive(self) -> Message:
        """Retrieve a pending message for a given user."""


class LocalCommunicator(Communicator):
    """Simulates a bidirectional star network with at most one message per 1-to-1 queue.

    The topology is a star: all workers communicate exclusively with the master.
    Workers cannot communicate with each other. Queues are implemented with
    collections.deque (not queue.Queue nor multiprocessing.Queue), so no thread locking is required.
    """

    def __init__(self, n_users: int, track_cc: bool = True) -> None:
        """Initialize the communicator and build message queues.

        Args:
            n_users (int): Total number of users (master + workers).
            track_cc (bool): Whether to track communication costs. Defaults to True.
        """
        self.n_users = n_users
        self.track_cc = track_cc

        self.n_workers = self.n_users - 1  # always one master
        self.queues = self._build_queues()

        if self.track_cc:
            self.cc_tracker = CommunicationCostTracker(n_workers=self.n_workers)
        return None

    def reset_queues(self) -> None:
        """Discard all pending messages by rebuilding the queue structure."""
        self.queues = self._build_queues()
        return None

    def _build_queues(self) -> dict[str, dict[str, deque] | deque]:
        """Create the inbox queues for all users.

        The master has one sub-queue per worker (keyed by worker id).
        Each worker has a single queue for messages from the master.

        Example:
            - Worker sending to master: message placed in queues[MASTER_ID][worker_id]
            - Master sending to worker_x: message placed in queues[worker_x]

        Returns:
            dict: Nested queue structure for all users.
        """
        q = {
            MASTER_ID: {
                f"{worker_id}": deque(maxlen=1) for worker_id in range(self.n_workers)
            },
        }
        for worker_id in range(self.n_workers):
            q[f"{worker_id}"] = deque(maxlen=1)
        return q

    def _check_empty_queue(self, src: str, to: str) -> None:
        """Raise an error if the recipient's queue already contains an unprocessed message.

        Args:
            src (str): Sender identifier.
            to (str): Recipient identifier.

        Raises:
            ValueError: If a previous message has not yet been consumed.
        """
        if to == "-1":
            if len(self.queues[to][src]) != 0:
                raise ValueError(
                    f"worker {src}, your previous message has not yet been processed by the master."
                )
        else:
            if len(self.queues[to]) != 0:
                raise ValueError(
                    f"master, your previous message has not yet been processed by the worker {to}."
                )
        return None

    def send_to_worker(self, to: str, subject: str, data: Any) -> None:
        """Place a message from the master into a worker's inbox queue.

        Args:
            to (str): Worker identifier.
            subject (str): Message subject.
            data (Any): Message payload.

        Raises:
            ValueError: If the worker id is not registered.
        """
        if to in self.queues[MASTER_ID].keys():
            msg = Message(src=MASTER_ID, to=to, subject=subject, data=data)
            self._check_empty_queue(src=MASTER_ID, to=to)
            self.queues[to].append(msg)
            if self.track_cc:
                self.cc_tracker.update(src=MASTER_ID, to=to, x=msg.weight)

        else:
            raise ValueError(f"{to=} is not present in the queue of the communicator.")
        return None

    def send_to_master(self, src: str, subject: str, data: Any) -> None:
        """Place a message from a worker into the master's inbox queue.

        Args:
            src (str): Worker identifier (sender).
            subject (str): Message subject.
            data (Any): Message payload.

        Raises:
            ValueError: If the master queue is not found.
        """
        if MASTER_ID in self.queues:
            msg = Message(src=src, to=MASTER_ID, subject=subject, data=data)
            self._check_empty_queue(src=src, to=MASTER_ID)
            self.queues[MASTER_ID][src].append(msg)
            if self.track_cc:
                self.cc_tracker.update(src=src, to=MASTER_ID, x=msg.weight)
        else:
            raise ValueError(
                f"{src=} is not present in the master_queue of the communicator."
            )
        return None

    def send(
        self, src: str, to: str | list[str], subject: str, data: Any
    ) -> None:
        """Add a message to the recipient's inbox and update the communication cost tracker.

        Routes the message to send_to_worker (when the master sends) or
        send_to_master (when a worker sends).

        Args:
            src (str): Sender identifier.
            to (str | list[str]): Recipient identifier or list of identifiers.
            subject (str): Message subject.
            data (Any): Message payload.

        Raises:
            TypeError: If to is neither a str nor a list of str.
        """
        if isinstance(to, str):
            ls_to = [to]
        elif isinstance(to, list):
            ls_to = to.copy()
        else:
            raise TypeError(
                f"expected type 'str' or 'list[str]' for parameter 'to', get '{type(to)}' instead."
            )

        # the master sends a message to one or many workers
        if src == "-1":
            for to_ in ls_to:
                self.send_to_worker(to=to_, subject=subject, data=data)

        # a worker sends a message to the master
        else:
            self.send_to_master(src=src, subject=subject, data=data)

        return None

    def receive(self, user_id: str, from_: Optional[str] = None) -> Optional[Message]:
        """Retrieve and remove the next pending message for a given user.

        For the master: retrieves from the sub-queue of the specified sender.
        For a worker: retrieves from its single incoming queue.

        Args:
            user_id (str): Identifier of the receiving user.
            from_ (Optional[str]): For the master, the worker id whose message to
                retrieve. Ignored for workers.

        Returns:
            Optional[Message]: The retrieved message, or None if the queue is empty.
        """
        # the master retrieves and removes ALL messages from every worker at the same time
        if user_id == MASTER_ID:

            # there is a message
            if self.queues[MASTER_ID][from_]:
                return self.queues[MASTER_ID][from_].popleft()

            # there is no message
            else:
                return None  # None means no message sent by the worker

        # the worker retrieves and removes the message of the master
        else:
            # there is a message
            if self.queues[user_id]:
                return self.queues[user_id].popleft()

            # there is no message
            else:
                return None
