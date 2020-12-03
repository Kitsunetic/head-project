import torch.nn as nn


def conv_block(inchannels, channels, kernel_size):
    p = kernel_size // 2

    return nn.Sequential(
        nn.Conv1d(inchannels, channels, kernel_size, padding=p),
        nn.BatchNorm1d(channels),
        nn.ReLU(inplace=True)
    )


class CNNBasedNet(nn.Module):
    def __init__(self):
        super(CNNBasedNet, self).__init__()

        self.conv = nn.Sequential(
            conv_block(6, 64, 15),
            conv_block(64, 128, 9),
            nn.AvgPool1d(2),
            conv_block(128, 256, 5),
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6 --> B, 6, S
        x = self.conv(x)
        return x
