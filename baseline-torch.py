import argparse
import numpy as np
import sys

import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from xqdm import xqdm

USING_COLS = ['timestamp',
              'acceleration_x', 'acceleration_y', 'acceleration_z',
              'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll',
              'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w']
X_COLS = ['acceleration_x', 'acceleration_y', 'acceleration_z',
          'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll',
          'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w',
          'input_orientation_xy', 'input_orientation_xz', 'input_orientation_xw',
          'input_orientation_yz', 'input_orientation_yw', 'input_orientation_zw']
Y_COLS = ['input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll']


def detect_collapse(csv):
    collapse_points = []

    timestamp = csv['timestamp']
    for i in range(len(csv) - 1):
        if timestamp[i + 1] - timestamp[i] > 11760000 * 6:
            # print(f'{i}\t{i + 1}\t{(timestamp[i + 1] - timestamp[i]) / 11760000}')
            collapse_points.append(i)
    return collapse_points


def interpolation(x, xt, y, yt, t):
    if xt == t:
        return x
    if yt == t:
        return y
    return x + (t - xt) / (yt - xt) * (y - x)


def dataset_interpolation(csv: pd.DataFrame):
    rows = {col: [float(csv[col][0])] for col in csv.columns}
    dt = 11760000  # 100/6 == 16.6666 ms in flicks
    nt = rows['timestamp'][0] + dt
    ni = 1

    while True:
        if ni >= len(csv) - 2:
            break

        for i in range(ni, len(csv)):
            if csv['timestamp'][i] > nt:
                for col in csv.columns:
                    intp = interpolation(csv[col][i - 1], csv['timestamp'][i - 1], csv[col][i], csv['timestamp'][i], nt)
                    rows[col].append(intp)
                break

        nt += dt
        ni = i - 1

    return pd.DataFrame(rows)


def dataset_clean(csv_file: Path):
    csv = pd.read_csv(csv_file)
    csv = csv[USING_COLS]
    csv['input_orientation_xy'] = csv['input_orientation_x'] * csv['input_orientation_y']
    csv['input_orientation_xz'] = csv['input_orientation_x'] * csv['input_orientation_z']
    csv['input_orientation_xw'] = csv['input_orientation_x'] * csv['input_orientation_w']
    csv['input_orientation_yz'] = csv['input_orientation_y'] * csv['input_orientation_z']
    csv['input_orientation_yw'] = csv['input_orientation_y'] * csv['input_orientation_w']
    csv['input_orientation_zw'] = csv['input_orientation_z'] * csv['input_orientation_w']

    # cut timestamps on collapse points
    collapse_points = detect_collapse(csv)
    if not collapse_points:
        csvs = [csv]
    elif len(collapse_points) == 1:
        if collapse_points[0] < 1500:
            # Throw head if collapse occurred before 2000idx
            csvs = [csv.iloc[collapse_points[0] + 1:]]
        else:
            csvs = [csv.iloc[:collapse_points[0]], csv.iloc[collapse_points[0] + 1:]]
    else:
        csvs = [csv.iloc[collapse_points[0] + 1:collapse_points[1]]]
        for i in range(1, len(collapse_points) - 1):
            csvs.append(csv.iloc[collapse_points[i] + 1:collapse_points[i + 1]])
        csvs.append(csv.iloc[collapse_points[-1] + 1:])

    csvs = [dataset_interpolation(csv) for csv in csvs]
    return csvs


def input_target_split(data: pd.DataFrame, x_cols, y_cols, upset=6, offset=6):
    X, Y = [], []
    for i in range(len(data) - upset - offset):
        x = []
        for j in range(i, i + upset):
            for col in x_cols:
                x.append(data.iloc[j][col])
        x = np.array(x)
        X.append(x)

        y = []
        for col in y_cols:
            y.append(data.iloc[i + upset + offset][col])
        y = np.array(y)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y


class SimpleDataset(Dataset):
    def __init__(self, csv_files, upset: int, offset: int):
        # clean datasets multiprocessing
        datasets = xqdm(dataset_clean, csv_files, desc='Loading datasets')

        # flatten dataset list in list type
        temp = []
        for data in datasets:
            for d in data:
                temp.append(d)
        datasets = temp

        # input/target split
        datasets = [input_target_split(data, X_COLS, Y_COLS, upset, offset) for data in datasets]

        # combine datasets
        X_train, X_valid, Y_train, Y_valid = [], [], [], []
        for x, y in datasets:
            # train/test split. 50% forward is train / backward is test
            assert x.shape[0] // 2 == y.shape[0] // 2
            split_idx = x.shape[0] // 2
            X_train.append(x[:split_idx])
            Y_train.append(y[:split_idx])
            X_valid.append(x[split_idx:])
            Y_valid.append(y[split_idx:])
        X_train = np.concatenate(X_train)
        Y_train = np.concatenate(Y_train)
        X_valid = np.concatenate(X_valid)
        Y_valid = np.concatenate(Y_valid)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

        self.fc = nn.Sequential(
            self._linear_block(36, 64),
            self._linear_block(64, 64),
            self._linear_block(64, 128),
            nn.Linear(128, 3)
        )

    def _linear_block(self, in_feats, out_feats):
        return nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def main(args):
    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='')
    args = p.parse_args(sys.argv[1:])
    main(args)
