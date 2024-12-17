""" Define Models """
import torch
import torch.nn as nn


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
