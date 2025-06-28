import torch
import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):
    def __init__(self, channel, res, output_dim):
        super().__init__()
        # c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 128 → 64
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64 → 32
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32 → 16
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        # compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, channel, res, res)
            conv_out = self.conv(dummy)
            conv_size = conv_out.shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(conv_size, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class ResNetEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, output_dim)

        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    def forward(self, x):
        return self.resnet(x)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
