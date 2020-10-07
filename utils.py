from math import pi

import torch

from torch_burn.metrics import Metric


def radian2degree(radian):
    return radian * 180 / pi


class YawMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        return radian2degree(torch.mean(torch.abs((outputs[:, 0] - targets[:, 0]) * self.std - self.mean)))


class PitchMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        return radian2degree(torch.mean(torch.abs((outputs[:, 1] - targets[:, 1]) * self.std - self.mean)))


class RollMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        return radian2degree(torch.mean(torch.abs((outputs[:, 2] - targets[:, 2]) * self.std - self.mean)))


class RMSMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        return radian2degree(torch.mean(torch.abs((outputs - targets) * self.std - self.mean)))
