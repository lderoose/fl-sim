import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistModel(nn.Module):
    INPUT_X = 28
    INPUT_Y = 28
    INPUT_SIZE = 784  # 28*28
    OUTPUT_SIZE = 10
    INPUT_CHANNEL = 1  # no RGB


class MnistModelSimple(MnistModel):
    def __init__(self) -> None:
        super(MnistModelSimple, self).__init__()
        self.fc = nn.Linear(self.INPUT_SIZE, self.OUTPUT_SIZE, bias=True)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.INPUT_SIZE)
        x = self.fc(x)
        return x


class MnistModelIntermediate(MnistModel):
    def __init__(self) -> None:
        super(MnistModelIntermediate, self).__init__()
        self.OUTPUT_CHANNEL = 1  # number of filters
        self.KERNEL_SIZE = 3  # size of the filter
        self.STRIDE = 1  # It defines the number of pixels whose convolution kernel moves at each step when the filter is applied to the input image.
        self.PADDING = 1  # rule for processing borders see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        self.conv1 = nn.Conv2d(
            in_channels=self.INPUT_CHANNEL,
            out_channels=self.OUTPUT_CHANNEL,
            kernel_size=self.KERNEL_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING,
        )
        self.fc1 = nn.Linear(self.INPUT_SIZE * self.OUTPUT_CHANNEL, 1024)
        self.fc2 = nn.Linear(1024, self.OUTPUT_SIZE)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.INPUT_X, self.INPUT_Y)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(
            x, kernel_size=3, stride=1, padding=1
        )  # aggregate output channel, often max or avg

        x = x.view(-1, self.INPUT_SIZE * self.OUTPUT_CHANNEL)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MnistLeNet5(MnistModel):
    def __init__(self):
        super(MnistLeNet5, self).__init__()
        self.OUTPUT_CHANNEL = 6

        self.conv1 = nn.Conv2d(
            in_channels=self.INPUT_CHANNEL,
            out_channels=self.OUTPUT_CHANNEL,
            kernel_size=5,
            padding=2,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.OUTPUT_SIZE)
        return None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.INPUT_CHANNEL, self.INPUT_X, self.INPUT_Y)
        x = F.relu(self.conv1(x))  # 28
        x = self.pool(x)  # 14
        x = F.relu(self.conv2(x))  # 10
        x = self.pool(x)  # 5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
