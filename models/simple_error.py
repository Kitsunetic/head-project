import torch.nn as nn


class SimpleError(nn.Module):
    def __init__(self):
        super(SimpleError, self).__init__()

        self.anynet = nn.Linear(1, 1)
        for p in self.anynet.parameters():
            p.requires_grad = False

    def forward(self, x):
        return x[:, 3:6, -1]
