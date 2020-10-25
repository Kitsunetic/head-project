from math import pi
from pathlib import Path
from typing import AnyStr

import numpy as np
import torch
import torch.nn as nn
import torch_burn as tb
from torch.utils.data import Dataset
from torch_burn.metrics import Metric


def radian2degree(radian):
    return radian * 180 / pi


class YawMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train: bool):
        return radian2degree(torch.mean(torch.abs((outputs[:, 0] - targets[:, 0]) * self.std)))


class PitchMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train: bool):
        return radian2degree(torch.mean(torch.abs((outputs[:, 1] - targets[:, 1]) * self.std)))


class RollMetric(Metric):
    def __init__(self, name: str, mode='min', mean=0, std=1):
        super().__init__(name, mode)
        self.mean = mean
        self.std = std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train: bool):
        return radian2degree(torch.mean(torch.abs((outputs[:, 2] - targets[:, 2]) * self.std)))


class RMSMetric(Metric):
    def __init__(self, name: str, yaw_std, pitch_std, roll_std, mode='min'):
        super().__init__(name, mode)
        self.yaw_std = yaw_std
        self.pitch_std = pitch_std
        self.roll_std = roll_std

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train: bool):
        yaw = radian2degree(torch.mean(torch.abs((outputs[:, 0] - targets[:, 0]) * self.yaw_std)))
        pitch = radian2degree(torch.mean(torch.abs((outputs[:, 1] - targets[:, 1]) * self.pitch_std)))
        roll = radian2degree(torch.mean(torch.abs((outputs[:, 2] - targets[:, 2]) * self.roll_std)))
        rms = torch.sqrt((yaw ** 2 + pitch ** 2 + roll ** 2) / 3)
        return rms


class SequentialDataset(Dataset):
    def __init__(self, xs, ys):
        super(SequentialDataset, self).__init__()

        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx].permute(1, 0)  # (6, 48; float32) --> sequence와 channel이 거꾸로라서 순서를 바꿔줌.
        y = self.ys[idx]  # (3, ; float32) --> 순서를 바꿔줄 필요가 없음...
        return x, y

    @staticmethod
    def load_data(data_path):
        data = torch.load(data_path)
        X_train, Y_train, X_test, Y_test = data['X_train'], data['Y_train'], data['X_test'], data['Y_test']

        train_ds = SequentialDataset(X_train, Y_train)
        test_ds = SequentialDataset(X_test, Y_test)
        return train_ds, test_ds


class HPSignalHistory(tb.metrics.InvisibleMetric):
    def __init__(self, filepath: AnyStr, name: str, verbose=True):
        super(HPSignalHistory, self).__init__(name=name, mode='min')

        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.on_valid = False
        self.target_signals = []
        self.out_signals = []

    def on_valid_epoch_begin(self, epoch: int):
        self.on_valid = True

        self.target_signals.clear()
        self.out_signals.clear()

    def on_valid_epoch_end(self, epoch: int, logs: dict):
        # 짧은 신호 단락을 3개의 신호 뭉치로 합치기
        target_signals = torch.cat(self.target_signals).transpose(0, 1)
        out_signals = torch.cat(self.out_signals).transpose(0, 1)

        # pickle 파일로 저장
        signals = torch.cat([target_signals, out_signals], dim=0).numpy()
        filepath = str(self.filepath).format(epoch=epoch, **logs)
        if self.verbose:
            print('Save output signals into', filepath)
        np.save(filepath, signals)

        # clear
        self.on_valid = False

        self.target_signals.clear()
        self.out_signals.clear()

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train: bool):
        if not is_train:
            self.target_signals.append(targets.detach().cpu())
            self.out_signals.append(outputs.detach().cpu())


class HPMetric(tb.metrics.InvisibleMetric):
    def __init__(self, name: str):
        super(HPMetric, self).__init__(name)

        self.diff = []

    def on_valid_epoch_end(self, epoch: int, logs: dict):
        # RMS, 99% tile 출력
        yaw_v, pitch_v, roll_v, rms_v, tile99_v = self._calc_values(self.diff)

        print(f'                  validation')
        print(f' - Yaw          : {yaw_v:10f}')
        print(f' - Pitch        : {pitch_v:10f}')
        print(f' - Roll         : {roll_v:10f}')
        print(f' - RMS          : {rms_v:10f}')
        print(f' - 99% Tile     : {tile99_v:10f}')

        self.diff.clear()

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train: bool):
        if not is_train:
            self.diff.append((outputs - targets).cpu())  # (B, 3)

    @staticmethod
    def _calc_values(diff):
        diff = torch.cat(diff).abs_()  # (B, 3)
        diff = radian2degree(diff)
        tile = diff.flatten().numpy()
        tile99 = np.percentile(tile, 99)

        mdiff = diff.mean(dim=0)
        rms = (mdiff.square().sum() / 3).sqrt()

        return mdiff[0].item(), mdiff[1].item(), mdiff[2].item(), rms.item(), tile99


class BaselineRNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=24, num_layers=8, dropout=0.0, bidirectional=False):
        super(BaselineRNN, self).__init__()

        # TODO 입력 전처리를 conv로 하는 것도?

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=bidirectional)

        out_channels = 2 * hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(out_channels, 3)

    def forward(self, x):
        outs, hiddens = self.rnn(x)
        x = outs[:, -1, ...]
        x = self.fc(x)

        return x


class BaselineGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=24, num_layers=8, dropout=0.0, bidirectional=False):
        super(BaselineGRU, self).__init__()

        # TODO 입력 전처리를 conv로 하는 것도?

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=bidirectional)

        out_channels = 2 * hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(out_channels, 3)

    def forward(self, x):
        outs, hiddens = self.rnn(x)
        x = outs[:, -1, ...]
        x = self.fc(x)

        return x


class BaselineLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=24, num_layers=8, dropout=0.0, bidirectional=False):
        super(BaselineLSTM, self).__init__()

        # TODO 입력 전처리를 conv로 하는 것도?

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=bidirectional)

        out_channels = 2 * hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(out_channels, 3)

    def forward(self, x):
        outs, (hiddens, cells) = self.rnn(x)
        x = outs[:, -1, ...]
        x = self.fc(x)

        return x
