import logging
import os
from abc import ABC
from typing import Any, Optional

import torch
from torch import nn

from conf.paths import LOG_DIR
from src import utils
from src.communication.communicator import Communicator
from src.communication.message import Message


class User(ABC):
    """Abstract base class for participants in the federated learning simulation.

    A User owns a model, communicates through a shared Communicator instance,
    and runs on a specific compute device. Concrete subclasses are Master and Worker.
    """

    def __init__(
        self,
        id: str,
        communicator: Communicator,
        device: str,
        log_level: Optional[str] = None,
    ) -> None:
        """Initialize a user.

        Args:
            id (str): Unique string identifier for this user.
            communicator (Communicator): Shared communicator used to exchange messages.
            device (str): Compute device string (e.g. "cpu" or "cuda:x").
            log_level (Optional[str]): Python logging level string. If None, logging
                is disabled for this user.
        """
        assert isinstance(
            id, str
        ), f"expected type 'str' for parameter 'id', get type {type(id)} instead."
        self.id = id
        self.communicator: Communicator = (
            communicator  # an instance shared instance with other users
        )
        self.device = device
        if log_level is not None:
            self._configureLogger(log_level)
        else:
            self.logger: Optional[logging.Logger] = None

        self.model: Optional[nn.Module] = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.id}"

    def _configureLogger(self, log_level: str) -> None:
        """Set up a file-based logger for this user.

        Args:
            log_level (str): Python logging level string (e.g. "DEBUG", "INFO").
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}{self.id}")
        log_file = LOG_DIR / f"{self.__class__.__name__}{self.id}.log"
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(log_level)
        return None

    def send_message(self, to: str, subject: str, data: Any) -> None:
        """Send a message to another user via the communicator.

        Args:
            to (str): Identifier of the recipient user.
            subject (str): Subject string describing the purpose of the message.
            data (Any): Payload to send.
        """
        self.communicator.send(src=self.id, to=to, subject=subject, data=data)
        return None

    def receive_message(self, from_: Optional[str] = None) -> Message:
        """Retrieve the next pending message from the communicator inbox.

        Args:
            from_ (Optional[str]): If provided, retrieve only the message from
                this specific sender. Only used by the master.

        Returns:
            Message: The retrieved message.
        """
        msg: Message = self.communicator.receive(self.id, from_)
        return msg

    def load_model(self, model: nn.Module) -> None:
        """Move the model to the user's device and cache gradient shapes.

        Args:
            model (nn.Module): PyTorch model to load.
        """
        self.model = model
        self.model.to(self.device)
        self.layer_gradient_shapes = utils.get_shape_gradients(self.model)
        return None
