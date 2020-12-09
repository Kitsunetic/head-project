import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_burn as tb


class HPMetric(tb.metrics.InvisibleMetric):
    def __init__(self, name: str, history_path, means=[0, 0, 0], stds=[1, 1, 1], logfile=None):
        super(HPMetric, self).__init__(name)

        self.means = means
        self.stds = stds

        self.diff = []
        self.train_history = defaultdict(list)
        self.valid_history = defaultdict(list)

        self.history_path = Path(history_path)
        self.history_path.mkdir(parents=True, exist_ok=True)

        self.logfile = None
        if logfile is not None:
            self.logfile = open(self.history_path / logfile, 'w')

    def on_train_epoch_end(self, epoch: int, logs: dict):
        # RMS, 99% tile 출력
        yaw_v, pitch_v, roll_v, rms_v, tile99_v = self._calc_values(self.diff)

        """
        print(f'                  validation')
        print(f' - Yaw          : {yaw_v:10f}')
        print(f' - Pitch        : {pitch_v:10f}')
        print(f' - Roll         : {roll_v:10f}')
        print(f' - RMS          : {rms_v:10f}')
        print(f' - 99% Tile     : {tile99_v:10f}')
        """

        self.train_history['yaw'].append(yaw_v)
        self.train_history['pitch'].append(pitch_v)
        self.train_history['roll'].append(roll_v)
        self.train_history['rms'].append(rms_v)
        self.train_history['tile99'].append(tile99_v)

        self.diff.clear()
        with open(self.history_path / 'train_history.pkl', 'wb') as f:
            pickle.dump(self.train_history, f)

    def on_valid_epoch_end(self, epoch: int, logs: dict):
        # RMS, 99% tile 출력
        yaw_v, pitch_v, roll_v, rms_v, tile99_v = self._calc_values(self.diff)

        print(f'                  validation')
        print(f' - Yaw          : {yaw_v:10f}')
        print(f' - Pitch        : {pitch_v:10f}')
        print(f' - Roll         : {roll_v:10f}')
        print(f' - RMS          : {rms_v:10f}')
        print(f' - 99% Tile     : {tile99_v:10f}')
        if self.logfile is not None:
            self.logfile.write(f'Epoch [{epoch:03d}] '
                               f'Yaw: {yaw_v:.4f}, Pitch: {pitch_v:.4f}, Roll: {roll_v:.4f}, '
                               f'RMS: {rms_v:.4f}, 99percentile {tile99_v:.4f}\n')

        self.valid_history['yaw'].append(yaw_v)
        self.valid_history['pitch'].append(pitch_v)
        self.valid_history['roll'].append(roll_v)
        self.valid_history['rms'].append(rms_v)
        self.valid_history['tile99'].append(tile99_v)

        self.diff.clear()
        with open(self.history_path / 'valid_history.pkl', 'wb') as f:
            pickle.dump(self.valid_history, f)

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train: bool):
        self.diff.append((outputs - targets).detach().cpu())  # (B, 3)

    def _calc_values(self, diff):
        diff = torch.cat(diff).abs_()  # (D, 3)
        for i in range(3):
            diff[:, i] *= self.stds[i]
        rms = (diff.square().sum(1) / 3).sqrt()
        tile = rms.flatten().numpy()
        tile99 = np.percentile(tile, 99)

        mrms = rms.mean()
        mdiff = diff.mean(dim=0)

        return mdiff[0].item(), mdiff[1].item(), mdiff[2].item(), mrms.item(), tile99
