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

    def forward(self, x):
        x = torch.flatten(x[:, :, :3], 1)  # B, 60, 3 --> B, 180
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


class MultiLayerPerceptron(nn.Module):
    expansion = 2

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(180 * self.expansion, 360 * self.expansion)
        self.layer2 = nn.Linear(360 * self.expansion, 720 * self.expansion)
        self.layer3 = nn.Linear(720 * self.expansion, 3)

    def forward(self, x):
        x = torch.flatten(x[:, :, :3], 1)  # B, 60, 3 --> B, 180
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x


class MLPBasedNet(nn.Module):
    expansion = 2

    def __init__(self):
        super(MLPBasedNet, self).__init__()

        self.linear_block = nn.Sequential(
            nn.Linear(180 * self.expansion, 360 * self.expansion),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(360 * self.expansion, 720 * self.expansion),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(720 * self.expansion, 720 * self.expansion),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(720 * self.expansion, 3)
        )

    def forward(self, x):
        x = torch.flatten(x[:, :, :3], 1)  # B, 60, 3 --> B, 180
        x = self.linear_block(x)
        return x
