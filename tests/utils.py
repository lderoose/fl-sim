import torch
import torch.nn as nn


class FakeTensor(torch.Tensor):
    """the class compressors need to deal with object having an attribute 'grad' filled"""

    def __init__(self, weight: torch.Tensor) -> None:
        self.grad = weight
        return None


class Model1kSingleLayer(nn.Module):

    def __init__(self) -> None:
        super(Model1kSingleLayer, self).__init__()
        self.fc = nn.Linear(1000, 1, bias=False)
        return None


class Model42k4Layers(nn.Module):

    def __init__(self) -> None:
        super(Model42k4Layers, self).__init__()
        self.fc = nn.ReLU()
        self.linear0 = nn.Linear(10, 100, bias=False)
        self.linear1 = nn.Linear(100, 200, bias=False)
        self.linear2 = nn.Linear(200, 100, bias=False)
        self.linear3 = nn.Linear(100, 10, bias=False)
        return None
