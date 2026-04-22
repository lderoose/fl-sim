import sys
from typing import Any

from src.communication.communication_cost import CommunicationCostComputer


class Message:
    """Represents a message exchanged between two users in the federated simulation.

    Stores the sender, recipient, subject, and data payload. The weight property
    computes the byte size of the payload based on the subject type.
    """

    def __init__(self, src: str, to: str, subject: str, data: Any) -> None:
        """Initialize a message.

        Args:
            src (str): Identifier of the sender.
            to (str): Identifier of the recipient.
            subject (str): Subject string describing the purpose of the message.
            data (Any): Payload carried by the message.
        """
        self.src = src
        self.to = to
        self.subject = subject
        self.data = data
        return None

    def __repr__(self) -> str:
        return f"message<src:{self.src}, to:{self.to}, subject: {self.subject}, data:{self.data}>"

    @property
    def weight(self) -> int:
        """Return the communication cost of this message in bytes, excluding the header.

        Returns:
            int: Number of bytes required to transmit the message payload.
        """
        n_bytes = 0

        if self.subject == "perform_one_step":
            # NB: we don't measure the communication cost associated to this message as its our internal stuff
            # Indeed, we send here the number of data the worker have to work on
            pass

        elif self.subject == "one_step_result":
            if isinstance(self.data, dict):
                for results in self.data.values():

                    tensor = results["gradient"]
                    cc_tensor = CommunicationCostComputer.compute_tensor_cc(tensor)
                    n_bytes += cc_tensor

                    if results["alpha"]:
                        cc_a = CommunicationCostComputer.get_n_bytes_from_precision(
                            precision="float64"
                        )
                        n_bytes += cc_a

                    # NB: we don't measure the communication cost associated to 'err' as its our internal stuff

            else:
                raise TypeError(
                    f"expected type dict for data attribute, get type '{type(self.data)}' instead."
                )

        elif self.subject == "update_weights":
            # the data attribute contains the full weight of the model or a compressed version, depending on the parameter master_weight_opt
            if isinstance(self.data, dict):
                for tensor in self.data.values():
                    cc_tensor = CommunicationCostComputer.compute_tensor_cc(tensor)
                    n_bytes += cc_tensor
            else:
                raise TypeError(
                    f"expected type dict for data attribute, get type '{type(self.data)}' instead."
                )

        elif self.subject == "test":
            if isinstance(self.data, str):
                n_bytes += sys.getsizeof(self.data)
            else:
                raise TypeError(
                    f"expected type str for data attribute, get type '{type(self.data)}' instead."
                )

        else:
            raise ValueError(
                f"define a rule to measure the communication cost of the message with subject '{self.subject}'"
            )

        return n_bytes
