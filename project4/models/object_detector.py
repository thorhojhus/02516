import torch
import torch.nn as nn
import torch.nn.functional as F

# Object Detecting Model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input: [batch, 3, 224, 224] # RGB 224x224 image
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),   # [batch, 8, 224, 224]
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),   # [batch, 8, 224, 224]
            nn.ReLU()
        )

        self.pooling = nn.MaxPool2d(
            kernel_size=2, stride=2                                # [batch, 8, 112, 112]
        )

        self.convolutional2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # [batch, 16, 112, 112]
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # [batch, 16, 112, 112]
            nn.ReLU()
        )

        self.pooling2 = nn.MaxPool2d(
            kernel_size=2, stride=2                                # [batch, 16, 56, 56]
        )

        self.convolutional3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # [batch, 32, 56, 56]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # [batch, 32, 56, 56]
            nn.ReLU()
        )

        self.pooling3 = nn.MaxPool2d(
            kernel_size=2, stride=2                                # [batch, 32, 28, 28]
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(32 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 2),  # background & potholes
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = self.pooling(x)
        x = self.convolutional2(x)
        x = self.pooling2(x)
        x = self.convolutional3(x)
        x = self.pooling3(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x