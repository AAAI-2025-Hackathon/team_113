import torch
import torch.nn as nn
import torch.nn.functional as F

class SOLUSConvNet(nn.Module):
    def __init__(self):
        super(SOLUSConvNet, self).__init__()
        
        # Convolutional layers with pooling to reduce size
        self.conv1 = nn.Conv2d(18, 32, kernel_size=3, stride=2, padding=1)  # Downsampling
        self.pool1 = nn.MaxPool2d(2, 2)  # MaxPooling to further downsample
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Downsampling
        self.pool2 = nn.MaxPool2d(2, 2)  # MaxPooling to further downsample
        
        # Global Average Pooling instead of flattening all features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Reduces each feature map to a single value

        # Final fully connected layer to map to 120x120 output
        self.fc1 = nn.Linear(64, 120 * 120)  # Output size is 120x120

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Apply Global Average Pooling (GAP)
        x = self.gap(x)  # Output shape: [batch_size, 64, 1, 1]
        
        # Flatten the output to feed it into the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64]
        
        # Fully connected layer to output the final result
        x = self.fc1(x)
        
        # Reshape to [batch_size, 120, 120]
        x = x.view(x.size(0), 120, 120)
        
        return x
        
class ERA5ConvNet(nn.Module):
    def __init__(self):
        super(ERA5ConvNet, self).__init__()
        
        # First convolutional layer: Input (28, 5, 5) -> Output (64, 3, 3)
        self.conv1 = nn.Conv2d(28, 64, kernel_size=3, stride=1, padding=1)  # Output: [28, 5, 5] -> [64, 5, 5]
        
        # Second convolutional layer: Input (64, 5, 5) -> Output (128, 3, 3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: [64, 5, 5] -> [128, 5, 5]
        
        # Fully connected layer: Flattened input (128 * 5 * 5) -> Output (120 * 120)
        self.fc1 = nn.Linear(128 * 5 * 5, 120 * 120)
        
    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the input to [batch_size, 128*5*5]
        
        # Pass through fully connected layer
        x = self.fc1(x)
        
        # Reshape to the desired output shape (120, 120)
        x = x.view(-1, 120, 120)  # Reshape to [batch_size, 120, 120]
        
        return x
        
class ElevationConvNet(nn.Module):
    def __init__(self):
        super(ElevationConvNet, self).__init__()
        
        # First Convolutional Layer with small kernel, small stride, and fewer channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=4, padding=1)  # Stride 4
        
        # Second Convolutional Layer with small kernel and fewer channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1)  # Stride 4
        
        # Pooling Layer: Max Pooling to reduce the spatial size
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce dimensions further
        
        # Third Convolutional Layer with reduced channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Stride 2
        
        # Calculate the flattened size after convolutions and pooling
        self.flattened_size = self._calculate_flattened_size(10812, 10812)
        
        # Smaller fully connected layer
        self.fc1 = nn.Linear(self.flattened_size, 256)  # Smaller layer size
        
        # Output layer (smaller)
        self.fc2 = nn.Linear(256, 120 * 120)  # Output size (120, 120)
    
    def _calculate_flattened_size(self, height, width):
        # Calculate the size of the output after the convolutions and pooling
        x = torch.randn(1, 1, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        return x.numel()  # Returns the number of elements (flattened size)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Adding a channel dimension if input is a 2D tensor
        
        # Apply convolutions and pooling
        x = self.pool1(self.conv3(self.conv2(self.conv1(x))))
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Reshape to match the final output size (120, 120)
        x = x.view(x.size(0), 120, 120)  # Reshaped to (120, 120)
        
        return x
        
class VegetationConvNet(nn.Module):
    def __init__(self):
        super(VegetationConvNet, self).__init__()

        # First convolutional layer: Reduce spatial dimensions and learn features
        self.conv1 = nn.Conv2d(15, 32, kernel_size=3, stride=2, padding=1)  # Output size: [32, 2176, 774]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output size: [64, 1088, 387]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output size: [128, 544, 194]

        # Global Average Pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: [128, 1, 1]

        # Fully connected layers
        self.fc1 = nn.Linear(128, 120 * 120)  # Flattened to size 128 (after pooling), output to 120x120
        self.fc2 = nn.Linear(120 * 120, 120 * 120)  # Output size: [120, 120]

    def forward(self, x):
        # Pass input through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Apply Global Average Pooling
        x = self.global_pool(x)

        # Flatten the output from the pooling layer
        x = torch.flatten(x, 1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape the output to [batch_size, 120, 120]
        x = x.view(-1, 120, 120)
        return x
        
class FCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        A simple encoder-decoder Fully Convolutional Network.
        Args:
          in_channels: number of input channels (here, 62)
          num_classes: number of output classes (here, 3 for labels 0, 1, 2)
        """
        super(FCN, self).__init__()
        # Encoder: three conv layers with pooling
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Decoder: upsampling via transpose convolutions
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))    # (64, H, W)
        x = self.pool1(x)                      # (64, H/2, W/2)

        x = F.relu(self.bn2(self.conv2(x)))    # (128, H/2, W/2)
        x = self.pool2(x)                      # (128, H/4, W/4)

        x = F.relu(self.bn3(self.conv3(x)))    # (256, H/4, W/4)
        x = self.pool3(x)                      # (256, H/8, W/8)

        # Decoder (upsample back to original resolution)
        x = self.upconv1(x)                    # (128, H/4, W/4)
        x = self.upconv2(x)                    # (64, H/2, W/2)
        x = self.upconv3(x)                    # (num_classes, H, W)
        return x
