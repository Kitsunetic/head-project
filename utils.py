import math
from dataclasses import dataclass

import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class History:
    epoch = 0
    loss = []

    yaw = []
    pitch = []
    roll = []
    rms = []



def radian2degree(radian):
    return radian * 180 / math.pi
