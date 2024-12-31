""" Define Models """
import torch
import torch.nn as nn
import torch.nn.init as init

# Simple Three-Layer Fully Connected Model
class ThreeLayerFC(nn.Module):
    def __init__(self):
        super(ThreeLayerFC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# LeNet-5 Convolutional Model
class PyTorchLeNet5(torch.nn.Module):
    def __init__(self, num_classes, grayscale=False):
        super().__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16*5*5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits

class SimpleCNNWithBatchNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNNWithBatchNorm, self).__init__()

        # Define convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for 32 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for 64 channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flattened input size: 64 feature maps of 14x14

        #self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(128, num_classes)  #Output size: 10 classes

    def forward(self, x):
        # Convolutional layers with BatchNorm, ReLU, and pooling
        x = self.pool(torch.relu(self.bn1(self.conv1(x)))) # Conv1 -> BatchNorm -> ReLU -> Pooling
        x = self.pool(torch.relu(self.bn2(self.conv2(x)))) # Conv2 -> BatchNorm -> ReLU -> Pooling

        # Flatten the feature maps for the fully connected layers
        x = x.view(x.size(0), -1)  #Flatten

        # Fully connected layers
        x = torch.relu(self.fc1(x))

        #x = self.dropout(x)
        x = self.fc2(x)  # No activation for the output layer (used for classification)

        return x

# A variant of AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(16, 48, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Adaptive pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

# A Convnet
class DeeperCIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeeperCIFARCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=16, num_channels=128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # New layer
        self.bn4 = nn.GroupNorm(num_groups= 32, num_channels=256)  # Batch normalization for new layer

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Adjusted for deeper network
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool2(torch.relu(self.bn4(self.conv4(x))))  # Forward pass for new layer

        # Flatten the feature maps
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x