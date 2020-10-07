import torch.nn as nn


class BaselineCNN1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaselineCNN1d, self).__init__()

        self.conv_in = nn.Conv1d()

    def forward(self, x):
        pass


