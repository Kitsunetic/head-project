import torch
import torch.nn as nn

from .resnet import ResBlock1d


class InceptionCRNNC(nn.Module):
    def __init__(self):
        super(InceptionCRNNC, self).__init__()

        self.rnn = nn.RNN(input_size=64,
                          hidden_size=128,
                          num_layers=4,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6
        outs, _ = self.rnn(x)  # B, S, 128
        x = outs.transpose(1, 2)  # B, C, S

        return x


class InceptionResNet1d(nn.Module):
    def __init__(self, block, layers, Activation=nn.LeakyReLU):
        super(InceptionResNet1d, self).__init__()

        self.inchannels = 64
        self.Activation = Activation
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(self.inchannels, channels, 3, stride=stride, Activation=self.Activation))
        self.inchannels = channels * block.channels
        for _ in range(1, blocks):
            layers.append(block(self.inchannels, channels, 3, Activation=self.Activation))

        return nn.Sequential(*layers)


class InceptionModel(nn.Module):
    def __init__(self):
        super(InceptionModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(16, 64, 7, padding=3, padding_mode='replicate', bias=False),
            nn.BatchNorm1d(64),
            nn.Hardswish(),
            ResBlock1d(64, 64, 3, Activation=nn.Hardswish),
            ResBlock1d(64, 64, 3, Activation=nn.Hardswish)
        )
        self.crnnc = InceptionCRNNC()
        self.resnet = InceptionResNet1d(ResBlock1d, [2, 2, 2])
        self.layer_out = nn.Sequential(
            ResBlock1d(512, 512, 3, Activation=nn.Hardswish),
            ResBlock1d(512, 1024, 3, Activation=nn.Hardswish),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),  # user가 여러명이므로 variance가 크기 때문에 dropout이 필요할지도?
            nn.Linear(1024, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        h1 = self.crnnc(x)  # B, 256, S
        h2 = self.resnet(x)  # B, 256, S
        x = torch.cat([h1, h2], dim=1)  # B, 512, S
        x = self.layer_out(x)

        return x
