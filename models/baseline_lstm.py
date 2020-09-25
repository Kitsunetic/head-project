import torch.nn as nn
from .utils import conv1d_bn_act


class BaselineCNN1D(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaselineCNN1D, self).__init__()

        activation = nn.ReLU()
        self.conv1 = conv1d_bn_act(1*input_size, 2*input_size, 32, activation)
        nn.MaxPool1d()
        self.conv2 = conv1d_bn_act(2*input_size, 2*input_size, 32, activation)
        self.conv2 = conv1d_bn_act(2*input_size, 2*input_size, 32, activation)
        self.conv2 = conv1d_bn_act(2*input_size, 2*input_size, 32, activation)
        self.conv2 = conv1d_bn_act(2*input_size, 2*input_size, 32, activation)

