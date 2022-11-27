import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFilter(nn.Module):
    ''' BN + ReLU + Convolution '''
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        if dropout == 0.0:
            self.filter = nn.Sequential(
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
                nn.Conv2d(in_features, out_features, 1, stride=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(),
                nn.Conv2d(out_features, out_features, 3, 1, padding=1)
            )
        else:
            self.filter = nn.Sequential(
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
                nn.Conv2d(in_features, out_features, 1, stride=1),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(out_features),
                nn.ReLU(),
                nn.Conv2d(out_features, out_features, 3, 1, padding=1),
                nn.Dropout2d(dropout)
            )

    def forward(self, xx):
        return self.filter(xx)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_features, growth_rate, dropout=0.0):
        super().__init__()
        self.layers = []
        for l in range(1, num_layers + 1):
            k = in_features + growth_rate * (l - 1)
            self.layers.append(DenseFilter(k, growth_rate, dropout=dropout))
        self.out_size = k
    
    def forward(self, xx):
        x = xx
        for l in range(len(self.layers)):
            x_tmp = self.layers[l](x)
            x = torch.cat((x, x_tmp), dim=1)
        return x


class DenseTransition(nn.Module):
    def __init__(self, in_features, compression_factor, dropout=0.0):
        super().__init__()
        self.out_size = torch.round(in_features * compression_factor)
        if dropout == 0.0:
            self.transition = nn.Sequential(
                nn.BatchNorm2d(in_features),
                nn.Conv2d(in_features, self.out_size, 1),
                nn.AvgPool2d(2, stride=2)
            )
        else:
            self.transition = nn.Sequential(
                nn.BatchNorm2d(in_features),
                nn.Conv2d(in_features, self.out_size, 1),
                nn.Dropout2d(dropout),
                nn.AvgPool2d(2, stride=2)
            )
        
    def forward(self, xx):
        return self.transition(xx)


class DenseNet121(nn.Module):
    def __init__(self, num_channels, num_classes, compression_factor=1.0, dropout=0.0):
        super().__init__()
        growth_rate = 32
        if compression_factor < 1.0:
            k = 2 * growth_rate
        else:
            k = growth_rate // 2
        # Convolution
        self.conv = nn.Conv2d(num_channels, k, 3, stride=2)
        # Pooling
        self.pool = nn.MaxPool2d(3, stride=2)
        # Dense Block (1)
        self.dense1 = DenseBlock(6, k, growth_rate, dropout=dropout)
        num_channels = self.dense1.out_size
        # Transition Layer (1)
        self.transition1 = DenseTransition(num_channels, compression_factor, dropout=dropout)
        num_channels = self.transition1.out_size
        # Dense Block (2)
        self.dense2 = DenseBlock(12, num_channels, growth_rate, dropout=dropout)
        num_channels = self.dense2.out_size
        # Transition Layer (2)
        self.transition2 = DenseTransition(num_channels, compression_factor, dropout=dropout)
        num_channels = self.transition2.out_size
        # Dense Block (3)
        self.dense3 = DenseBlock(24, num_channels, growth_rate, dropout=dropout)
        num_channels = self.dense3.out_size
        # Transition Layer (3)
        self.transition3 = DenseTransition(num_channels, compression_factor, dropout=dropout)
        num_channels = self.transition3.out_size
        # Dense Block (4)
        self.dense4 = DenseBlock(16, num_channels, growth_rate, dropout=dropout)
        num_channels = self.dense3.out_size
        # Classification Layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(num_channels, num_classes)
        )
    
    def forward(self, xx):
        x = self.conv(xx)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.dense3(x)
        x = self.transition3(x)
        x = self.dense4(x)
        return self.classifier(x)
