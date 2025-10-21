import torch, torch.nn as nn, torch.nn.functional as F, torchvision.models as models

class ConvNet(nn.Module):
    """Similar to the ConvNet used in DualStream paper"""
    def __init__(self, in_channels=3, out_dim=4096):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.full6 = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.full7 = nn.Sequential(
            nn.Linear(4096, out_dim),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.full6(x.view(x.size(0), -1))
        x = self.full7(x)
        return x


class DualStreamNet(nn.Module):
    """Dual Stream Network combining RGB and Optical Flow streams"""
    def __init__(self, num_flow_frames=9, intermediate_dim=2048, num_classes=10):
        super(DualStreamNet, self).__init__()
        self.spatial_stream = ConvNet(in_channels=3, out_dim=intermediate_dim)
        self.temporal_stream = ConvNet(in_channels=2*num_flow_frames, out_dim=intermediate_dim)
        self.classifier = nn.Linear(intermediate_dim * 2, num_classes)

    def forward(self, x_spatial, x_temporal):
        # x_spatial: [B, 3, H, W], x_temporal: [B, T, 2, H, W]
        rgb_out = self.spatial_stream(x_spatial)

        batch_size = x_temporal.size(0)
        x_temporal = x_temporal.view(batch_size, -1, x_temporal.size(3), x_temporal.size(4))
        flow_out = self.temporal_stream(x_temporal)

        combined = torch.cat((rgb_out, flow_out), dim=1)
        out = self.classifier(combined)
        return out


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

class DualStreamResNet(nn.Module):
    """Dual Stream Network combining RGB and Optical Flow streams"""
    def __init__(self, num_flow_frames=9, num_classes=10):
        super(DualStreamResNet, self).__init__()

        self.spatial_stream = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        intermediate_dim = self.spatial_stream.fc.in_features
        self.spatial_stream.fc = nn.Identity()
        
        self.temporal_stream = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.temporal_stream.fc = nn.Identity()
        self.temporal_stream = mod_resnet_inputs(self.temporal_stream, in_channels=2 * num_flow_frames)
        
        # freeze
        for param in self.spatial_stream.parameters():
            param.requires_grad = False
        for param in self.temporal_stream.parameters():
            param.requires_grad = False

        for param in self.temporal_stream.conv1.parameters():
            param.requires_grad = True

        for param in self.spatial_stream.layer4.parameters():
            param.requires_grad = True
        for param in self.temporal_stream.layer4.parameters():
            param.requires_grad = True
        
        self.classifier = nn.Linear(intermediate_dim * 2, num_classes)

    def forward(self, x_spatial, x_temporal):
        # x_spatial: [B, 3, H, W], x_temporal: [B, T, 2, H, W]
        rgb_out = self.spatial_stream(x_spatial)

        batch_size = x_temporal.size(0)
        x_temporal = x_temporal.view(batch_size, -1, x_temporal.size(3), x_temporal.size(4))
        flow_out = self.temporal_stream(x_temporal)

        combined = torch.cat((rgb_out, flow_out), dim=1)
        out = self.classifier(combined)
        return out


if __name__ == "__main__":
    model = DualStreamNet(num_flow_frames=9, intermediate_dim=2048, num_classes=10)
    rgb_input = torch.randn(8, 3, 224, 224)
    flow_input = torch.randn(8, 9, 2, 224, 224)  # 9 frames of optical flow
    output = model(rgb_input, flow_input)
    print(output.shape)