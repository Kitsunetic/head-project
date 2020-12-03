from .conv import CNNBasedNet
from .crnnc import CLSTM, CLSTMC, CRNNC, CGRUC
from .mlp import SingleLayerPerceptron, SecondLayerPerceptron, MultiLayerPerceptron, MLPBasedNet
from .resnet import ResNet1d
from .rnn import BidirectionalStackedRNN, BidirectionalStackedGRU, BidirectionalStackedLSTM
from .rnn import StackedRNN, StackedGRU, StackedLSTM


def _make_map(*networks):
    nets = {}
    for net in networks:
        nets[net.__class__.__name__] = net

    return nets


_network_map = _make_map(SingleLayerPerceptron, SecondLayerPerceptron, MultiLayerPerceptron, MLPBasedNet,
                         CNNBasedNet,
                         StackedRNN, StackedGRU, StackedLSTM,
                         BidirectionalStackedRNN, BidirectionalStackedGRU, BidirectionalStackedLSTM,
                         ResNet1d,
                         CLSTM, CLSTMC, CRNNC, CGRUC)


def get_model_by_name(name, *args, **kwargs):
    net = _network_map[name]
    return net(*args, **kwargs)
