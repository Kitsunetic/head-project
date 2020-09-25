import torch.nn as nn


class FullyConnectedModel(nn.Sequential):
    def __init__(self, input_size, output_size):
        super(FullyConnectedModel, self).__init__()
        self.add_module('fc1', nn.Linear(input_size, output_size))
