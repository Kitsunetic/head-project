import torch.nn as nn


class StackedRNN(nn.Module):
    def __init__(self, bidirectional=False):
        super(StackedRNN, self).__init__()

        self.rnn = nn.RNN(input_size=6, hidden_size=64, num_layers=8, dropout=0.2, bidirectional=bidirectional)
        self.fc = nn.Linear(128 if bidirectional else 64, 3)

    def forward(self, x):
        outs, _ = self.rnn(x)
        x = outs[:, -1, :]  # B, 60, 64 --> B, 64
        x = self.fc(x)
        return x


class StackedGRU(nn.Module):
    def __init__(self, bidirectional=False):
        super(StackedGRU, self).__init__()

        self.rnn = nn.GRU(input_size=6, hidden_size=64, num_layers=8, dropout=0.2,
                          batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(128 if bidirectional else 64, 3)

    def forward(self, x):
        outs, _ = self.rnn(x)
        x = outs[:, -1, :]  # B, 60, 64 --> B, 64
        x = self.fc(x)
        return x


class StackedLSTM(nn.Module):
    def __init__(self, bidirectional=False):
        super(StackedLSTM, self).__init__()

        self.rnn = nn.GRU(input_size=6, hidden_size=64, num_layers=8, dropout=0.2,
                          batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(128 if bidirectional else 64, 3)

    def forward(self, x):
        outs, _ = self.rnn(x)
        x = outs[:, -1, :]  # B, 60, 64 --> B, 64
        x = self.fc(x)
        return x


class BidirectionalStackedRNN(StackedRNN):
    def __init__(self):
        super(BidirectionalStackedRNN, self).__init__(bidirectional=True)


class BidirectionalStackedGRU(StackedGRU):
    def __init__(self):
        super(BidirectionalStackedGRU, self).__init__(bidirectional=True)


class BidirectionalStackedLSTM(StackedLSTM):
    def __init__(self):
        super(BidirectionalStackedLSTM, self).__init__(bidirectional=True)
