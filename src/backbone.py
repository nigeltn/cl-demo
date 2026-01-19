import torch
import torch.nn as nn
from torchvision.models import resnet18


class SplitResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base_model = resnet18(weights=None)

        base_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        base_model.maxpool = nn.Identity()

        self.G = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
        )

        self.F = nn.Sequential(
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
            nn.Flatten(),
        )

        self.fc = nn.Linear(512, num_classes)

    def forward_G(self, x):
        return self.G(x)

    def forward_F(self, z):
        z = self.F(z)
        return self.fc(z)

    def forward(self, x):
        z = self.G(x)
        z = self.F(z)
        return self.fc(z)
