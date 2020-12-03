import torch
import torch.nn as nn


class SingleLayerPerceptron(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.layer = nn.Linear(360, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)  # B, 60, 6 --> B, 360
        x = self.layer(x)
        return x


class SecondLayerPerceptron(nn.Module):
    def __init__(self):
        super(SecondLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(360, 360)
        self.layer2 = nn.Linear(360, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)  # B, 60, 6 --> B, 360
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(360, 360)
        self.layer2 = nn.Linear(360, 360)
        self.layer3 = nn.Linear(360, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)  # B, 60, 6 --> B, 360
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MLPBasedNet(nn.Module):
    def __init__(self):
        super(MLPBasedNet, self).__init__()

        self.linear_block = nn.Sequential(
            nn.Linear(360, 360),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(360, 360),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(360, 360),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(360, 3)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)  # B, 60, 6 --> B, 360
        x = self.linear_block(x)
        return x
