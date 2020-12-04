import torch.nn as nn

from .resnet import ResBlock1d, ResBlock1dPReLU


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
    def __init__(self):
        super(CLSTMC, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            ResBlock1d(64, 64, 3)
        )

        self.rnn = nn.LSTM(input_size=64,
                           hidden_size=64,
                           num_layers=4,
                           batch_first=True,
                           dropout=0,
                           bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 128, 3, stride=2),
            ResBlock1d(128, 256, 3, stride=2)
        )
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
    def __init__(self):
        super(CRNNC, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            ResBlock1d(64, 64, 3)
        )

        self.rnn = nn.RNN(input_size=64,
                          hidden_size=64,
                          num_layers=4,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 128, 3, stride=2),
            ResBlock1d(128, 256, 3, stride=2)
        )
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
    def __init__(self):
        super(CGRUC, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            ResBlock1d(64, 64, 3)
        )

        self.rnn = nn.GRU(input_size=64,
                          hidden_size=64,
                          num_layers=4,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 128, 3, stride=2),
            ResBlock1d(128, 256, 3, stride=2)
        )
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


class CLSTMCFC(nn.Module):
    def __init__(self):
        super(CLSTMCFC, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            ResBlock1d(64, 64, 3)
        )

        self.rnn = nn.LSTM(input_size=64,
                           hidden_size=64,
                           num_layers=6,
                           batch_first=True,
                           dropout=0.2,
                           bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 128, 3, stride=2),
            ResBlock1d(128, 256, 3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
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


class CRNNCFCPReLU(nn.Module):
    def __init__(self):
        super(CRNNCFCPReLU, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            ResBlock1dPReLU(64, 128, 3, stride=2),
            ResBlock1dPReLU(128, 256, 3, stride=2)
        )

        self.rnn = nn.RNN(input_size=256,
                          hidden_size=128,
                          num_layers=6,
                          batch_first=True,
                          dropout=0.2,
                          bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1dPReLU(128, 256, 3, stride=2),
            ResBlock1dPReLU(256, 512, 3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(0.2),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.2),
            nn.PReLU(),
            nn.Linear(1024, 3)
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
