import torch.nn as nn
import torch.nn.functional as F
import torch

class Base_CNN(nn.Module):
    def __init__(self):
        super(Base_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.dropout = nn.Dropout(0.3)

        self.classifier1 = nn.Linear(256, 64)
        self.classifier2 = nn.Linear(64, 6)

        self.residualConv = nn.Conv2d(1, 64, kernel_size=1, stride=2, padding=1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        residual = self.residualConv(residual)
        x = F.pad(x, (0, 1))
        residual = residual[:, :, :x.shape[2], :x.shape[3]]
        x = x + residual

        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier1(x)
        x = self.classifier2(x)

        return x
