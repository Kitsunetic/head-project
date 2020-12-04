import torch.nn as nn

from .conv import CNNBasedNet
from .crnnc import CGRUC, CLSTM, CLSTMC, CRNNC
from .mlp import MLPBasedNet, MultiLayerPerceptron, SecondLayerPerceptron, SingleLayerPerceptron
from .resnet import ResNet1d
from .rnn import BidirectionalStackedGRU, BidirectionalStackedLSTM, BidirectionalStackedRNN
from .rnn import StackedGRU, StackedLSTM, StackedRNN


def _make_map(*networks):
    nets = {}
    for net in networks:
        nets[net.__name__] = net

    return nets


_network_map = _make_map(SingleLayerPerceptron, SecondLayerPerceptron, MultiLayerPerceptron, MLPBasedNet,
                         CNNBasedNet,
                         StackedRNN, StackedGRU, StackedLSTM,
                         BidirectionalStackedRNN, BidirectionalStackedGRU, BidirectionalStackedLSTM,
                         ResNet1d,
                         CLSTM, CLSTMC, CRNNC, CGRUC)


def get_model_by_name(name, *args, **kwargs) -> nn.Module:
    assert name in _network_map, 'Network must be one of ' + str(list(_network_map.keys()))

    net = _network_map[name]
    return net(*args, **kwargs)
