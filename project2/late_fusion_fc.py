import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
in_features = model.fc.in_features
out_features = 10

class fc(nn.Module):
    def __init__(self, in_features, out_features):
        super(fc, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
model.fc = fc(in_features, out_features)

class LateFusionNetwork(nn.Module):
    def __init__(self, num_frames=10, num_classes=10, feature_dim=16):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(num_frames * feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
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

late_fusion_model = LateFusionNetwork(num_frames=10, num_classes=10, feature_dim=2048)