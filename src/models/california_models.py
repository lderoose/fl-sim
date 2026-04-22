import torch
import torch.nn as nn


class CaliforniaModel(nn.Module):
    INPUT_SIZE = 8
    OUTPUT_SIZE = 1


class CaliforniaModelSimple(CaliforniaModel):
    """linear regression"""

    def __init__(self) -> None:
        super(CaliforniaModelSimple, self).__init__()
        self.fc = nn.Linear(self.INPUT_SIZE, self.OUTPUT_SIZE, bias=True)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CaliforniaModelIntermediate(CaliforniaModel):
    """MLP"""

    def __init__(self) -> None:
        super(CaliforniaModelIntermediate, self).__init__()
        self.fc1 = nn.Linear(8, 64, bias=False)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1, bias=False)
        self.relu = nn.ReLU()
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
