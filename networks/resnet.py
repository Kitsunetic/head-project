import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inchannels, channels, kernel_size, stride=1, groups=1, Activation=nn.LeakyReLU):
        super(ResBlock1d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannels, channels, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=groups, padding_mode='replicate'),
            nn.BatchNorm1d(channels),
            Activation(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=groups, padding_mode='replicate'),
            nn.BatchNorm1d(channels)
        )
        self.act = Activation()

        self.conv2 = None
        if inchannels != channels or stride != 1:
            self.conv2 = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride, groups=groups),
                nn.BatchNorm1d(channels)
            )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        if self.conv2 is not None:
            identity = self.conv2(identity)
        x += identity
        x = self.act(x)

        return x


class ResBlock1dPReLU(nn.Module):
    expansion = 1

    def __init__(self, inchannels, channels, kernel_size, stride=1, groups=1):
        super(ResBlock1dPReLU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannels, channels, kernel_size, padding=kernel_size // 2, stride=stride, groups=groups),
            nn.BatchNorm1d(channels),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm1d(channels)
        )
        self.act = nn.LeakyReLU()

        self.conv2 = None
        if inchannels != channels:
            self.conv2 = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride, groups=groups),
                nn.BatchNorm1d(channels)
            )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        if self.conv2 is not None:
            identity = self.conv2(identity)
        x += identity
        x = self.act(x)

        return x


class ResNet1d(nn.Module):
    def __init__(self, block, layers):
        super(ResNet1d, self).__init__()

        self.inchannels = 64

        self.conv = nn.Sequential(
            nn.Conv1d(6, self.inchannels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inchannels),
            nn.LeakyReLU()
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(512 * block.channels, 3)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6 --> B, 6, S
        x = self.conv(x)  # B, C, S

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(self.inchannels, channels, 3, stride=stride))
        self.inchannels = channels * block.channels
        for _ in range(1, blocks):
            layers.append(block(self.inchannels, channels, 3))

        return nn.Sequential(*layers)


class ResNet15(ResNet1d):
    def __init__(self):
        super(ResNet15, self).__init__(ResBlock1d, [2, 2, 2, 2])
