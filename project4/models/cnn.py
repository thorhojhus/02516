import torch
import torch.nn as nn

# Simple encoder-decoder segmentation CNN

class ConvBlock(nn.Module):
    """Two conv2d blocks"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class SegmentationCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_channels=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels * 2)
        self.up0 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(base_channels, base_channels)

        # Output head
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))

        # Bottleneck
        x3 = self.bottleneck(self.pool2(x2))

        # Decoder
        x = self.up1(x3)
        x = self.dec1(x)
        x = self.up0(x)
        x = self.dec0(x)

        logits = self.head(x)
        return logits
