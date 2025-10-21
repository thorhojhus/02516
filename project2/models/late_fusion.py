import torch
import torch.nn as nn
import torchvision.models as models

class LateFusionMLP(nn.Module):
    def __init__(self, num_frames=10, num_classes=10, feature_dim=2048, hidden_dim=256):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(num_frames * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Input: [B, T, C, H, W]
        # FrameVideoDataset: [B, C, T, H, W]
        
        if x.shape[1] == 3:  # check if [B, C, T, H, W]
            
            # to [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()  
        
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        
        with torch.no_grad():
            # [B*T, feature_dim, 1, 1]
            features = self.feature_extractor(x)  

        # [B, T, feature_dim]
        features = features.view(b, t, -1)
        
        # [B, T*feature_dim]
        features = features.flatten(1)
        
        # [B, num_classes]
        out = self.mlp(features)
        return out


class LateFusionPool(nn.Module):
    def __init__(self, num_classes=10, feature_dim=256):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # Input: [B, T, C, H, W]
        # FrameVideoDataset: [B, C, T, H, W]
        
        if x.shape[1] == 3:  # check if [B, C, T, H, W]
            
            # to [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()  
        
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        
        with torch.no_grad():
            # [B*T, feature_dim, 1, 1]
            features = self.feature_extractor(x)  

        # [B, T, feature_dim]
        features = features.view(b, t, -1)
        
        pooled_features = features.mean(dim=1)
        
        out = self.classifier(pooled_features)
        return out