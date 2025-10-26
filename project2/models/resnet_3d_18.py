import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models.video as models

class ResNet3D18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet3D18, self).__init__()
        self.model = models.r3d_18(weights=models.R3D_18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)