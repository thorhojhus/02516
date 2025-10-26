import torch
import torch.nn as nn
import torchvision.models as models

class PerFrameModel(nn.Module):
    def __init__(self):
        super(PerFrameModel, self).__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        out_features = 10
        model.fc = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(in_features, out_features)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)