import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
in_features = model.fc.in_features
out_features = 10 # 101  # Number of classes in UCF101

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