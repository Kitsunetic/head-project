import torch.nn as nn


class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x[:, :3, :].flatten(1)  # batch랑 yaw, pitch, roll만 남기고 flatten
        x = self.fc(x)
        return x


class BaselineFC2(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaselineFC2, self).__init__()

        self.fc = nn.ModuleList([
            nn.Linear(input_size, input_size),
            nn.Linear(input_size, output_size)
        ])
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.act(x)
        x = self.fc[-1](x)
        return x


class BaselineFC3(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaselineFC3, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 2 * input_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * input_size, 2 * input_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * input_size, 4 * input_size),
            nn.ReLU(inplace=True),
            nn.Linear(4 * input_size, 4 * input_size),
            nn.ReLU(inplace=True),
            nn.Linear(4 * input_size, 8 * input_size),
            nn.ReLU(inplace=True),
            nn.Linear(8 * input_size, output_size)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
