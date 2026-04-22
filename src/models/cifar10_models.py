import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import BasicBlock, ResNet


class CIFAR10Model(nn.Module):
    INPUT_X = 32
    INPUT_Y = 32
    INPUT_SIZE = 1024  # 32*32
    OUTPUT_SIZE = 10
    INPUT_CHANNEL = 3  # no RGB


class CIFAR10ModelSimple(CIFAR10Model):
    def __init__(self) -> None:
        super(CIFAR10ModelSimple, self).__init__()
        self.fc = nn.Linear(
            self.INPUT_CHANNEL * self.INPUT_X * self.INPUT_Y,
            self.OUTPUT_SIZE,
            bias=True,
        )
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.INPUT_CHANNEL * self.INPUT_X * self.INPUT_Y)
        x = self.fc(x)
        return x


class CIFAR10ModelIntermediate(CIFAR10Model):
    def __init__(self) -> None:
        super(CIFAR10ModelIntermediate, self).__init__()
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
        reshaping_dim = (
            batch_size,
            self.INPUT_CHANNEL,
            self.INPUT_SIZE,
            self.INPUT_SIZE,
        )
        x = x.reshape(reshaping_dim)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(
            x, kernel_size=3, stride=1, padding=1
        )  # aggregate output channel, often max or avg

        x = x.view(-1, self.INPUT_SIZE * self.OUTPUT_CHANNEL)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR10LeNet5(CIFAR10Model):
    def __init__(self):
        super(CIFAR10LeNet5, self).__init__()
        self.OUTPUT_CHANNEL = 6

        self.conv1 = nn.Conv2d(
            in_channels=self.INPUT_CHANNEL,
            out_channels=self.OUTPUT_CHANNEL,
            kernel_size=5,
            padding=2,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.OUTPUT_SIZE)
        return None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.INPUT_CHANNEL, self.INPUT_X, self.INPUT_Y)
        x = F.relu(self.conv1(x))  # 32
        x = self.pool(x)  # 16
        x = F.relu(self.conv2(x))  # 12
        x = self.pool(x)  # 6
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# https://arxiv.org/pdf/1608.06037
# Lets Keep it simple, Using simple architectures to outperform deeper and more complex architectures (2016).
class CIFAR10SimpleNet(CIFAR10Model):
    def __init__(self):
        super(CIFAR10SimpleNet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                self.INPUT_CHANNEL, self.INPUT_X, kernel_size=3, padding=1
            ),  # 32x32x32
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 64x8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 128x4x4
        )

        self.fc_block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.OUTPUT_SIZE),
        )
        return None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.INPUT_CHANNEL, self.INPUT_X, self.INPUT_Y)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x


class CIFAR10ResNet20(ResNet):
    def __init__(self) -> None:
        super(CIFAR10ResNet20, self).__init__(
            block=BasicBlock, num_blocks=[3, 3, 3], num_classes=CIFAR10Model.OUTPUT_SIZE
        )
        return None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(
            batch_size,
            CIFAR10Model.INPUT_CHANNEL,
            CIFAR10Model.INPUT_X,
            CIFAR10Model.INPUT_Y,
        )
        return super().forward(x)
