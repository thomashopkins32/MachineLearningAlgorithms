import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
            self.project = nn.Conv2d(in_channels, out_channels, 1, stride=2)
            self.bnp = nn.BatchNorm2d(out_channels)
        elif in_channels != out_channels:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.project = nn.Conv2d(in_channels, out_channels, 1)
            self.bnp = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
            self.project = None
            self.bnp = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, xx):
        x = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        if self.project:
            xx = self.project(xx)
        return F.relu(x + xx)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bot_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, bot_channels, 1, stride=2)
            self.project = nn.Conv2d(in_channels, out_channels, 1, stride=2)
            self.bnp = nn.BatchNorm2d(out_channels)
        elif in_channels != out_channels:
            self.conv1 = nn.Conv2d(in_channels, bot_channels, 1)
            self.project = nn.Conv2d(in_channels, out_channels, 1)
            self.bnp = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, bot_channels, 3, padding=1)
            self.project = None
            self.bnp = None
        self.bn1 = nn.BatchNorm2d(bot_channels)
        self.conv2 = nn.Conv2d(bot_channels, bot_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bot_channels)
        self.conv3 = nn.Conv2d(bot_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, xx):
        x = F.relu(self.bn1(self.conv1(xx)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.project:
            xx = self.bnp(self.project(xx))
        return F.relu(x + xx)
    
class ResNet18(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # conv2_x
            nn.MaxPool2d(3, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            
            # conv3_x
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128),

            # conv4_x
            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 256),

            # conv5_x
            ResidualBlock(256, 512, downsample=True),
            ResidualBlock(512, 512),

            # output
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(512, 1000))
    
    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # conv2_x
            nn.MaxPool2d(3, stride=2),
            BottleneckBlock(64, 64, 256),
            BottleneckBlock(256, 64, 256),
            BottleneckBlock(256, 64, 256),
            
            # conv3_x
            BottleneckBlock(256, 128, 512, downsample=True),
            BottleneckBlock(512, 128, 512),
            BottleneckBlock(512, 128, 512),
            BottleneckBlock(512, 128, 512),

            # conv4_x
            BottleneckBlock(512, 256, 1024, downsample=True),
            BottleneckBlock(1024, 256, 1024),
            BottleneckBlock(1024, 256, 1024),
            BottleneckBlock(1024, 256, 1024),
            BottleneckBlock(1024, 256, 1024),
            BottleneckBlock(1024, 256, 1024),

            # conv5_x
            BottleneckBlock(1024, 512, 2048, downsample=True),
            BottleneckBlock(2048, 512, 2048),
            BottleneckBlock(2048, 512, 2048),

            # output
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(2048, 1000))
    
    def forward(self, x):
        return self.model(x)