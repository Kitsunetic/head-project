from math import pi
import numpy as np

import torch

from torch_burn.metrics import Metric, InvisibleMetric


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


class HeadProjectMetric(InvisibleMetric):
    def __init__(self, name: str):
        super(HeadProjectMetric, self).__init__(name)

        self.diff = []
        self.is_train = True

    def on_valid_epoch_begin(self):
        self.is_train = False

    def on_valid_epoch_end(self):
        # RMS, 99% tile 출력
        diff = torch.cat(self.diff)  # (B, 3, W)
        tile = diff.flatten().numpy()
        tile99_1 = np.percentile(tile, 99)

        bdiff = diff.mean(dim=2)
        tile = bdiff.flatten().numpy()
        tile99_2 = np.percentile(tile, 99)

        mdiff = bdiff.mean(dim=0)
        rms = (mdiff.square().sum() / 3).sqrt()

        print(' - Yaw         :', mdiff[0].item())
        print(' - Pitch       :', mdiff[1].item())
        print(' - Roll        :', mdiff[2].item())
        print(' - RMS         :', rms.item())
        print(' - 99% Tile_1  :', tile99_1)
        print(' - 99% Tile_2  :', tile99_2)

        self.is_train = False
        self.diff.clear()

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        if self.is_train:
            return

        self.diff.append((outputs - targets).cpu())  # (B, 3)
