import torch.nn as nn
import torch.nn.functional as F


class FemnistModel(nn.Module):
    INPUT_X = 28
    INPUT_Y = 28
    INPUT_SIZE = 784  # 28*28
    INPUT_CHANNEL = 1  # no RGB
    NUM_CLASSES = 62
            

class FemnistCNN(FemnistModel):
    def __init__(self):
        super(FemnistCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.INPUT_CHANNEL, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(7*7*64, 2048)
        self.fc2 = nn.Linear(2048, self.NUM_CLASSES)

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # pool_size=2, stride=2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # pool_size=2, stride=2
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class FemnistLeNet5(FemnistModel):
    def __init__(self):
        super(FemnistLeNet5, self).__init__()
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
        self.fc3 = nn.Linear(84, self.NUM_CLASSES)
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
    
    
class FemnistLeNet5BN(FemnistModel):
    def __init__(self):
        super(FemnistLeNet5BN, self).__init__()
        self.OUTPUT_CHANNEL1 = 6
        self.OUTPUT_CHANNEL2 = 16

        # Convolution + BatchNorm
        self.conv1 = nn.Conv2d(
            in_channels=self.INPUT_CHANNEL,
            out_channels=self.OUTPUT_CHANNEL1,
            kernel_size=5,
            padding=2
        )
        self.bn1 = nn.BatchNorm2d(self.OUTPUT_CHANNEL1, track_running_stats=False)

        self.conv2 = nn.Conv2d(
            in_channels=self.OUTPUT_CHANNEL1,
            out_channels=self.OUTPUT_CHANNEL2,
            kernel_size=5,
            padding=0
        )
        self.bn2 = nn.BatchNorm2d(self.OUTPUT_CHANNEL2, track_running_stats=False)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(self.OUTPUT_CHANNEL2 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.NUM_CLASSES)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        return None

    def forward(self, x):
        # Ensure input shape [batch_size, channels, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.INPUT_CHANNEL, self.INPUT_X, self.INPUT_Y)

        # Conv1 -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv2 -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
