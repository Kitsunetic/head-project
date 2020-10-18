from math import pi

import numpy as np
import pandas as pd
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

        self.diff_t, self.diff_v = [], []
        self.is_train = True

        self.yaw_t, self.pitch_t, self.roll_t, self.rms_t, self.tile99_t = [], [], [], [], []
        self.yaw_v, self.pitch_v, self.roll_v, self.rms_v, self.tile99_v = [], [], [], [], []

    def on_train_epoch_begin(self):
        self.is_train = True

    def on_valid_epoch_begin(self):
        self.is_train = False

    def on_valid_epoch_end(self):
        # RMS, 99% tile 출력
        yaw_t, pitch_t, roll_t, rms_t, tile99_t = self._calc_values(self.diff_t)
        yaw_v, pitch_v, roll_v, rms_v, tile99_v = self._calc_values(self.diff_v)

        # memorize score history for future plotting
        self.yaw_t.append(yaw_t)
        self.pitch_t.append(pitch_t)
        self.roll_t.append(roll_t)
        self.rms_t.append(rms_t)
        self.tile99_t.append(tile99_t)
        self.yaw_v.append(yaw_v)
        self.pitch_v.append(pitch_v)
        self.roll_v.append(roll_v)
        self.rms_v.append(rms_v)
        self.tile99_v.append(tile99_v)

        print(f'                  train      validation')
        print(f' - Yaw          : {yaw_t:10f} {yaw_v:10f}')
        print(f' - Pitch        : {pitch_t:10f} {pitch_v:10f}')
        print(f' - Roll         : {roll_t:10f} {roll_v:10f}')
        print(f' - RMS          : {rms_t:10f} {rms_v:10f}')
        print(f' - 99% Tile     : {tile99_t:10f} {tile99_v:10f}')
        print(f' - Min RMS / 99%: {min(self.rms_t):10f} {min(self.rms_v):10f} '
              f'{min(self.tile99_t):10f} {min(self.tile99_v):10f}')

        self.diff_t.clear()
        self.diff_v.clear()

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        if self.is_train:
            self.diff_t.append((outputs - targets).cpu())  # (B, 3)
        else:
            self.diff_v.append((outputs - targets).cpu())  # (B, 3)

    def to_csv(self, filename):
        data = {'yaw_t': self.yaw_t, 'pitch_t': self.pitch_t, 'roll_t': self.roll_t,
                'rms_t': self.rms_t, 'tile99_t': self.tile99_t,
                'yaw_v': self.yaw_v, 'pitch_v': self.pitch_v, 'roll_v': self.roll_v,
                'rms_v': self.rms_v, 'tile99_v': self.tile99_v}
        data = pd.DataFrame(data)
        data.to_csv(filename, index=False)

    @staticmethod
    def _calc_values(diff):
        diff = torch.cat(diff).abs_()  # (B, 3)
        diff = radian2degree(diff)
        tile = diff.flatten().numpy()
        tile99 = np.percentile(tile, 99)

        mdiff = diff.mean(dim=0)
        rms = (mdiff.square().sum() / 3).sqrt()

        return mdiff[0].item(), mdiff[1].item(), mdiff[2].item(), rms.item(), tile99
