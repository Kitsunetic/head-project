import torch
import torch.nn as nn
from .resnet import ResBlock1d


class InceptionCRNNC(nn.Module):
    def __init__(self):
        super(InceptionCRNNC, self).__init__()
        Activation = nn.Hardswish

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 5, padding=2, groups=2, padding_mode='replicate'),
            nn.BatchNorm1d(64),
            Activation(),
            ResBlock1d(64, 64, 3, Activation=Activation),
            ResBlock1d(64, 128, 3, Activation=Activation),
            ResBlock1d(128, 128, 3, Activation=Activation)
        )

        self.rnn = nn.RNN(input_size=128,
                          hidden_size=128,
                          num_layers=4,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)

        self.conv_out = nn.Sequential(
            ResBlock1d(256, 256, 3, Activation=Activation),
            ResBlock1d(256, 256, 3, Activation=Activation)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6 --> B, 6, S
        x = self.conv_in(x)  # B, 6, S
        x = x.transpose(1, 2)  # B, S, 6

        outs, _ = self.rnn(x)  # B, S, 128
        x = outs.transpose(1, 2)  # B, C, S
        x = self.conv_out(x)  # B, C, S

        return x


class InceptionConvModule(nn.Module):
    def __init__(self):
        super(InceptionConvModule, self).__init__()



    def forward(self, x):
        pass


class InceptionModel(nn.Module):
    def __init__(self):
        super(InceptionModel, self).__init__()

        self.crnnc = InceptionCRNNC()

    def forward(self, x):
        pass
