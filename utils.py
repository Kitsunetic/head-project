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
        return radian2degree(torch.mean(torch.abs((outputs[:, 0] - targets[:, 0]) * self.std)))


class PitchMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        return radian2degree(torch.mean(torch.abs((outputs[:, 1] - targets[:, 1]) * self.std)))


class RollMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        return radian2degree(torch.mean(torch.abs((outputs[:, 2] - targets[:, 2]) * self.std)))


class RMSMetric(Metric):
    def __init__(self, name: str, yaw_std, pitch_std, roll_std, mode='min'):
        super().__init__(name, mode)
        self.yaw_std = yaw_std
        self.pitch_std = pitch_std
        self.roll_std = roll_std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        yaw = radian2degree(torch.mean(torch.abs((outputs[:, 0] - targets[:, 0]) * self.yaw_std)))
        pitch = radian2degree(torch.mean(torch.abs((outputs[:, 1] - targets[:, 1]) * self.pitch_std)))
        roll = radian2degree(torch.mean(torch.abs((outputs[:, 2] - targets[:, 2]) * self.roll_std)))
        rms = torch.sqrt((yaw ** 2 + pitch ** 2 + roll ** 2) / 3)
        return rms
