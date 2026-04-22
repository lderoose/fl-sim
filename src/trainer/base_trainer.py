from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator


class Trainer(ABC):
    """Abstract base class for all trainers.

    Subclasses must implement the training loop logic via `context_epoch` and `__call__`.
    """

    @abstractmethod
    @contextmanager
    def context_epoch(self, epoch: int) -> Generator[None, None, None]:
        """Context manager wrapping the body of a single training epoch.

        Args:
            epoch (int): Index of the current epoch.
        """

    @abstractmethod
    def __call__(self) -> None:
        """Run the full training procedure."""
