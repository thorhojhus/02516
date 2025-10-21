import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

def mod_resnet_inputs(model, in_channels):
    """Modify the first conv layer of ResNet to accept different number of input channels"""
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=original_conv.out_channels,
                            kernel_size=original_conv.kernel_size,
                            stride=original_conv.stride,
                            padding=original_conv.padding,
                            bias=original_conv.bias is not None)
    with torch.no_grad():
        if in_channels > 3:
            model.conv1.weight[:, :3, :, :] = original_conv.weight
            for i in range(3, in_channels):
                model.conv1.weight[:, i:i+1, :, :] = original_conv.weight[:, :1, :, :]
        else:
            model.conv1.weight[:, :in_channels, :, :] = original_conv.weight[:, :in_channels, :, :]
    return model

class EarlyFusionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EarlyFusionCNN, self).__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone = mod_resnet_inputs(backbone, in_channels=3 * 10)  # assuming 10 frames, each with 3 channels
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        for param in backbone.conv1.parameters():
            param.requires_grad = True

        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):

        if x.shape[1] == 3:  # check if [B, C, T, H, W]
            # to [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()  

        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size, C*num_frames, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size, -1)
        # fused_features = torch.mean(features, dim=1)
        out = self.classifier(features)
        return out
