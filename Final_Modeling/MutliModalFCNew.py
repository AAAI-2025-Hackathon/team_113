import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

# 1. Elevation Data: Input assumed to be (batch, 1, 10812, 10812)
#    Expected output: (batch, 1, 120, 120)
class ElevationNet(nn.Module):
    def __init__(self):
        super(ElevationNet, self).__init__()
        self.conv = nn.Sequential(
            # Use a larger stride to quickly reduce the huge input resolution.
            nn.Conv2d(1, 8, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Map back to one channel.
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Adaptive pooling will output (120,120) regardless of the spatial size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((120, 120))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        return x

# 2. Vegetation Data: Input shape (batch, 15, 4353, 1547)
#    Expected output: (batch, 15, 120, 120)
class VegetationNet(nn.Module):
    def __init__(self):
        super(VegetationNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            # Map back to 15 channels.
            nn.Conv2d(32, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((120, 120))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        return x

# 3. Soil Variable Data: Input shape (batch, 28, 5, 5) with batch=1096
#    Expected output: (batch, 28, 120, 120)
#    Because the spatial size is very small, we use upsampling.
class SoilVariableNet(nn.Module):
    def __init__(self):
        super(SoilVariableNet, self).__init__()
        self.conv = nn.Sequential(
            # A simple convolutional block that preserves the 28 channels.
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Use bilinear upsampling to scale from 5x5 to 120x120.
        self.upsample = nn.Upsample(size=(120, 120), mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

# 4. Soil Composition Data: Input shape (batch, 18, 1306, 464)
#    Expected output: (batch, 18, 120, 120)
class SoilCompositionNet(nn.Module):
    def __init__(self):
        super(SoilCompositionNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            # Map back to 18 channels.
            nn.Conv2d(32, 18, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((120, 120))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
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
