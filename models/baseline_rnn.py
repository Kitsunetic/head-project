import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, return_all_layers=False):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x.size(0))

        print(x.shape, h.shape)
        x, h = self.rnn(x, h)  # (B, seq_length, hidden_size)

        print(x.shape, h.shape)

        if self.return_all_layers:
            pass
        return x

    def init_hidden(self, batch_size):
        device = next(self.rnn.parameters())[0].device
        return torch.zeros(batch_size, self.num_layers, self.hidden_size).to(device)


class RNNBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNBaseline, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        # x: (B, 6, 48)
        x = self.rnn(x)
        x = self.fc(x)
        return x
