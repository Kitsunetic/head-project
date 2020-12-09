import torch.nn as nn

from .conv import CNNBasedNet
from .crnnc import CGRUC, CLSTM, CLSTMC, CRNNC
from .crnnc import CGRUC_Hardswish, CLSTMC_Hardswish, CRNNC_Hardswish
from .crnnc import CGRUC_Hardswish_FC, CLSTMC_Hardswish_FC, CRNNC_Hardswish_FC
from .crnnc import CRNNC5, CRNNC_PReLU, CRNNC_ReLU, CRNNC_Tanh
from .mlp import MLPBasedNet, MultiLayerPerceptron, SecondLayerPerceptron, SingleLayerPerceptron
from .resnet import ResNet15
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
                         ResNet15,
                         CLSTM, CLSTMC, CRNNC, CGRUC,
                         CRNNC_PReLU, CRNNC_ReLU, CRNNC_Hardswish, CRNNC_Tanh,
                         CLSTMC_Hardswish, CGRUC_Hardswish, CRNNC_Hardswish,
                         CLSTMC_Hardswish_FC, CGRUC_Hardswish_FC, CRNNC_Hardswish_FC,
                         CRNNC5)


def get_model_by_name(name, *args, **kwargs) -> nn.Module:
    assert name in _network_map, 'Network must be one of ' + str(list(_network_map.keys()))

    net = _network_map[name]
    return net(*args, **kwargs)
