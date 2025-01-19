""" Define Models """
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ThreeLayerFCNorm(nn.Module): 
    def __init__(self): 
        super(ThreeLayerFCNorm, self).__init__() 
        self.fc1 = nn.Linear(28 * 28, 128) 
        self.gn1 = nn.GroupNorm(8, 128)  # Group normalization with 4 groups 
        self.fc2 = nn.Linear(128, 64) 
        self.gn2 = nn.GroupNorm(4, 64)  # Group normalization with 4 groups 
        self.fc3 = nn.Linear(64, 10) 
 
    def forward(self, x): 
        x = x.view(-1, 28 * 28) 
        x = self.fc1(x) 
        x = self.gn1(x)  # Apply group normalization 
        x = torch.relu(x) 
        x = self.fc2(x) 
        x = self.gn2(x)  # Apply group normalization 
        x = torch.relu(x) 
        return self.fc3(x)

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

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, num_groups=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.norm1 = norm_layer(out_channels) if norm_layer != nn.GroupNorm else norm_layer(num_groups, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = norm_layer(out_channels) if norm_layer != nn.GroupNorm else norm_layer(num_groups, out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, norm_type="group", num_groups=32):
        super(ResNet, self).__init__()
        self.in_channels = 16

        if norm_type == "group":
            norm_layer = lambda channels: nn.GroupNorm(min(num_groups, channels), channels)
        elif norm_type == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "layer":
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm_layer(16) if norm_layer != nn.GroupNorm else norm_layer(num_groups, 16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion) if norm_layer != nn.GroupNorm else nn.GroupNorm(num_groups=32, num_channels=out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, norm_layer))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Instantiate ResNet18
def ResNet18(num_classes=10, norm_type="group", num_groups=32):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, norm_type=norm_type, num_groups=num_groups)

# Instantiate ResNet20
class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Layers
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class VGG11(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(VGG11, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights if required
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)           # Pass through feature extractor
        x = self.avgpool(x)            # Apply adaptive average pooling
        x = torch.flatten(x, 1)        # Flatten the tensor
        x = self.classifier(x)         # Pass through classifier
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
