import torch.nn as nn


def conv_block(inchannels, channels, kernel_size):
    p = kernel_size // 2

    return nn.Sequential(
        nn.Conv1d(inchannels, channels, kernel_size, padding=p, padding_mode='replicate'),
        nn.BatchNorm1d(channels),
        nn.Hardswish(inplace=True)
    )


class CNNBasedNet(nn.Module):
    def __init__(self):
        super(CNNBasedNet, self).__init__()

        self.conv = nn.Sequential(
            conv_block(6, 64, 5),
            conv_block(64, 128, 3),
            nn.AvgPool1d(2),
            conv_block(128, 256, 3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),

            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.Hardswish(inplace=True),
            nn.Linear(512, 1024),
            nn.Dropout(0.2),
            nn.Hardswish(inplace=True),
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # B, 60, 6 --> B, 6, 60
        x = self.conv(x)
        return x
