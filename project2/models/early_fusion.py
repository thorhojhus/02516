import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class EarlyFusionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EarlyFusionCNN, self).__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):

        if x.shape[1] == 3:  # check if [B, C, T, H, W]
            # to [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()  

        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_frames, -1)
        fused_features = torch.mean(features, dim=1)
        out = self.classifier(fused_features)
        return out