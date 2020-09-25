import torch.nn as nn


def conv1d_bn_act(in_channels, out_channels, kernel_size, act):
    valid_padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=valid_padding),
        act(inplace=True),
        nn.BatchNorm1d(out_channels)
    )
