import torch
import torch.nn as nn


def cba(in_channels, out_channels, kernel_size):
    p = kernel_size // 2

    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=p),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )


class BaselineCNN1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaselineCNN1d, self).__init__()

        self.conv_in = cba(in_channels, 64, 15)

        self.conv = nn.Sequential(
            cba(64, 128, 9),
            nn.AvgPool1d(2),
            cba(128, 256, 5)
        )

        self.out_pool = nn.AdaptiveAvgPool1d(1)
        self.out_fc = nn.Linear(256, out_channels)

    def forward(self, x):
        x = self.conv_in(x)  # B, 128, 48
        x = self.conv(x)  # B, 128, 12
        x = self.out_pool(x)  # B, 128, 1
        x = torch.flatten(x, 1)  # B, 128
        x = self.out_fc(x)  # B, 3
        return x
