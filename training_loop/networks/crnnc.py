import torch.nn as nn

from .resnet import ResBlock1d


class CLSTM(nn.Module):
    def __init__(self):
        super(CLSTM, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 24, 3, padding=1, groups=2),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
            nn.Conv1d(24, 48, 3, padding=1, groups=2),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.Conv1d(48, 64, 3, padding=1),
            nn.BatchNorm1d(64),
        )

        self.rnn = nn.LSTM(input_size=64,
                           hidden_size=64,
                           num_layers=8,
                           batch_first=True,
                           dropout=0,
                           bidirectional=True)

        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, 48, 6 --> B, 6, 48
        x = self.conv_in(x)  # B, 64, 48
        x = x.transpose(1, 2)  # B, 48, 64

        outs, _ = self.rnn(x)
        x = outs[:, -1, :]
        x = self.fc(x)

        return x


class CLSTMC(nn.Module):
    def __init__(self, Activation=nn.LeakyReLU, big_fc=False):
        super(CLSTMC, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2, padding_mode='replicate'),
            nn.BatchNorm1d(64),
            Activation(),
            ResBlock1d(64, 64, 3, Activation=Activation)
        )

        self.rnn = nn.LSTM(input_size=64,
                           hidden_size=64,
                           num_layers=4,
                           batch_first=True,
                           dropout=0,
                           bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 128, 3, stride=2, Activation=Activation),
            ResBlock1d(128, 256, 3, stride=2, Activation=Activation)
        )
        if big_fc:
            self.fc = nn.Sequential(
                nn.Linear(256, 512),
                nn.Dropout(0.2),
                Activation(),
                nn.Linear(512, 512),
                nn.Dropout(0.2),
                Activation(),
                nn.Linear(512, 3)
            )
        else:
            self.fc = nn.Linear(256, 3)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6 --> B, 6, S
        x = self.conv_in(x)  # B, 6, S
        x = x.transpose(1, 2)  # B, S, 6

        outs, _ = self.rnn(x)  # B, S, 128
        x = outs.transpose(1, 2)  # B, C, S
        x = self.conv_out(x)  # B, C, S
        x = x[:, :, -1]  # B, C
        x = self.fc(x)  # B, 3

        return x


class CRNNC(nn.Module):
    def __init__(self, Activation=nn.LeakyReLU, big_fc=False):
        super(CRNNC, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2, padding_mode='replicate'),
            nn.BatchNorm1d(64),
            Activation(),
            ResBlock1d(64, 64, 3, Activation=Activation)
        )

        self.rnn = nn.RNN(input_size=64,
                          hidden_size=64,
                          num_layers=4,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 128, 3, stride=2, Activation=Activation),
            ResBlock1d(128, 256, 3, stride=2, Activation=Activation)
        )
        if big_fc:
            self.fc = nn.Sequential(
                nn.Linear(256, 512),
                nn.Dropout(0.2),
                Activation(),
                nn.Linear(512, 512),
                nn.Dropout(0.2),
                Activation(),
                nn.Linear(512, 3)
            )
        else:
            self.fc = nn.Linear(256, 3)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6 --> B, 6, S
        x = self.conv_in(x)  # B, 6, S
        x = x.transpose(1, 2)  # B, S, 6

        outs, _ = self.rnn(x)  # B, S, 128
        x = outs.transpose(1, 2)  # B, C, S
        x = self.conv_out(x)  # B, C, S
        x = x[:, :, -1]  # B, C
        x = self.fc(x)  # B, 3

        return x


class CGRUC(nn.Module):
    def __init__(self, Activation=nn.LeakyReLU, big_fc=False):
        super(CGRUC, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2, padding_mode='replicate'),
            nn.BatchNorm1d(64),
            Activation(),
            ResBlock1d(64, 64, 3, Activation=Activation)
        )

        self.rnn = nn.GRU(input_size=64,
                          hidden_size=64,
                          num_layers=4,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 128, 3, stride=2, Activation=Activation),
            ResBlock1d(128, 256, 3, stride=2, Activation=Activation)
        )
        if big_fc:
            self.fc = nn.Sequential(
                nn.Linear(256, 512),
                nn.Dropout(0.2),
                Activation(),
                nn.Linear(512, 512),
                nn.Dropout(0.2),
                Activation(),
                nn.Linear(512, 3)
            )
        else:
            self.fc = nn.Linear(256, 3)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6 --> B, 6, S
        x = self.conv_in(x)  # B, 6, S
        x = x.transpose(1, 2)  # B, S, 6

        outs, _ = self.rnn(x)  # B, S, 128
        x = outs.transpose(1, 2)  # B, C, S
        x = self.conv_out(x)  # B, C, S
        x = x[:, :, -1]  # B, C
        x = self.fc(x)  # B, 3

        return x


class CRNNC5(nn.Module):
    def __init__(self):
        super(CRNNC5, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 5, padding=2, groups=2, padding_mode='replicate'),
            nn.BatchNorm1d(64),
            nn.Hardswish(),
            ResBlock1d(64, 64, 3, Activation=nn.Hardswish),
            ResBlock1d(64, 128, 3, Activation=nn.Hardswish),
            ResBlock1d(128, 128, 3, Activation=nn.Hardswish),
            ResBlock1d(128, 256, 3, Activation=nn.Hardswish),
            ResBlock1d(256, 256, 3, Activation=nn.Hardswish)
        )

        self.rnn = nn.RNN(input_size=256,
                          hidden_size=256,
                          num_layers=8,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(256, 512, 3, stride=2, Activation=nn.Hardswish),
            ResBlock1d(512, 512, 3, stride=2, Activation=nn.Hardswish)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(0.2),
            nn.Hardswish(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.Hardswish(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6 --> B, 6, S
        x = self.conv_in(x)  # B, 6, S
        x = x.transpose(1, 2)  # B, S, 6

        outs, _ = self.rnn(x)  # B, S, 128
        x = outs.transpose(1, 2)  # B, C, S
        x = self.conv_out(x)  # B, C, S
        x = x[:, :, -1]  # B, C
        x = self.fc(x)  # B, 3

        return x


class CRNNC_PReLU(CRNNC):
    def __init__(self):
        super(CRNNC_PReLU, self).__init__(nn.PReLU)


class CRNNC_ReLU(CRNNC):
    def __init__(self):
        super(CRNNC_ReLU, self).__init__(nn.ReLU)


class CRNNC_Tanh(CRNNC):
    def __init__(self):
        super(CRNNC_Tanh, self).__init__(nn.Tanh)


class CRNNC_Hardswish(CRNNC):
    def __init__(self):
        super(CRNNC_Hardswish, self).__init__(nn.Hardswish)


class CGRUC_Hardswish(CGRUC):
    def __init__(self):
        super(CGRUC_Hardswish, self).__init__(nn.Hardswish)


class CLSTMC_Hardswish(CLSTMC):
    def __init__(self):
        super(CLSTMC_Hardswish, self).__init__(nn.Hardswish)


class CRNNC_Hardswish_FC(CRNNC):
    def __init__(self):
        super(CRNNC_Hardswish_FC, self).__init__(nn.Hardswish, big_fc=True)


class CGRUC_Hardswish_FC(CGRUC):
    def __init__(self):
        super(CGRUC_Hardswish_FC, self).__init__(nn.Hardswish, big_fc=True)


class CLSTMC_Hardswish_FC(CLSTMC):
    def __init__(self):
        super(CLSTMC_Hardswish_FC, self).__init__(nn.Hardswish, big_fc=True)
