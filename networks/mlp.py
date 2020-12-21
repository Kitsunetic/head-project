import torch
import torch.nn as nn


class SingleLayerPerceptron(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.layer = nn.Linear(180, 3)

    def forward(self, x):
        x = torch.flatten(x[:, :, :3], 1)  # B, 60, 3 --> B, 180
        x = self.layer(x)
        return x


class SecondLayerPerceptron(nn.Module):
    def __init__(self):
        super(SecondLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(180, 360)
        self.layer2 = nn.Linear(360, 3)
        self.activation = nn.Hardswish()

    def forward(self, x):
        x = torch.flatten(x[:, :, :3], 1)  # B, 60, 3 --> B, 180
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class MultiLayerPerceptron(nn.Module):
    expansion = 2

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(180 * self.expansion, 360 * self.expansion)
        self.layer2 = nn.Linear(360 * self.expansion, 720 * self.expansion)
        self.layer3 = nn.Linear(720 * self.expansion, 3)
        self.activation = nn.Hardswish()

    def forward(self, x):
        x = torch.flatten(x[:, :, :3], 1)  # B, 60, 3 --> B, 180
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return x


class MLPBasedNet(nn.Module):
    def __init__(self):
        super(MLPBasedNet, self).__init__()

        channels = 6
        sequence = 120
        Q = channels * sequence
        self.linear_block = nn.Sequential(
            nn.Linear(Q, 2 * Q),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(2 * Q, 4 * Q),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(4 * Q, 4 * Q),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(4 * Q, 3)
        )

    def forward(self, x):
        x = torch.flatten(x[:, :, :6], 1)
        x = self.linear_block(x)
        return x
