import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from xqdm import xqdm

USING_COLS = [
    'timestamp',
    # 'acceleration_x', 'acceleration_y', 'acceleration_z',
    'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll',
    # 'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w'
]
X_COLS = [
    # 'acceleration_x', 'acceleration_y', 'acceleration_z',
    'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll',
    # 'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w',
    # 'input_orientation_xy', 'input_orientation_xz', 'input_orientation_xw',
    # 'input_orientation_yz', 'input_orientation_yw', 'input_orientation_zw'
]
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


def dataset_interpolation(data):
    # rows = {col: [float(csv[col][0])] for col in csv.columns}
    rows = {k: [] for k in data.keys()}
    dt = 11760000  # 100/6 == 16.6666 ms in flicks
    nt = data['timestamp'][0] + dt
    ni = 1

    while True:
        if ni >= len(data['timestamp']) - 2:
            break

        for i in range(ni, len(data['timestamp'])):
            if data['timestamp'][i] > nt:
                for col in data.keys():
                    intp = interpolation(data[col][i - 1],
                                         data['timestamp'][i - 1],
                                         data[col][i],
                                         data['timestamp'][i],
                                         nt)
                    rows[col].append(intp)
                break

        nt += dt
        ni = i - 1

    return pd.DataFrame(rows)


def dataset_clean(csv_file: Path):
    csv = pd.read_csv(csv_file)
    csv = csv[USING_COLS]
    # csv['input_orientation_xy'] = csv['input_orientation_x'] * csv['input_orientation_y']
    # csv['input_orientation_xz'] = csv['input_orientation_x'] * csv['input_orientation_z']
    # csv['input_orientation_xw'] = csv['input_orientation_x'] * csv['input_orientation_w']
    # csv['input_orientation_yz'] = csv['input_orientation_y'] * csv['input_orientation_z']
    # csv['input_orientation_yw'] = csv['input_orientation_y'] * csv['input_orientation_w']
    # csv['input_orientation_zw'] = csv['input_orientation_z'] * csv['input_orientation_w']

    # cut timestamps on collapse points
    collapse_points = detect_collapse(csv)
    if not collapse_points:
        csvs = [csv]
    elif len(collapse_points) == 1:
        if collapse_points[0] < 3000:
            # Throw head if collapse occurred before 2000idx
            csvs = [csv.iloc[collapse_points[0] + 1:]]
        else:
            csvs = [csv.iloc[:collapse_points[0]], csv.iloc[collapse_points[0] + 1:]]
    else:
        csvs = [csv.iloc[collapse_points[0] + 1:collapse_points[1]]]
        for i in range(1, len(collapse_points) - 1):
            csvs.append(csv.iloc[collapse_points[i] + 1:collapse_points[i + 1]])
        csvs.append(csv.iloc[collapse_points[-1] + 1:])

    datas = [{col: c[col].to_numpy(dtype=np.float32) for col in csv.columns} for c in csvs]
    datas = [dataset_interpolation(data) for data in datas]
    return datas


def input_target_split(data, x_cols, y_cols, upset=6, offset=6, hop=6):
    # print('len data', len(data))
    X, Y = [], []
    for i in range(0, len(data) - upset - offset, hop):
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


def __input_target_split(args):
    return input_target_split(*args)


def make_dataset(csv_files, upset: int, offset: int, hop: int):
    # clean datasets multiprocessing
    datasets = xqdm(dataset_clean, csv_files, desc='Loading datasets', ncols=100)

    # flatten dataset list in list type
    temp = []
    for data in datasets:
        for d in data:
            if len(d) > 10000:
                temp.append(d)
    datasets = temp

    # input/target split
    items = [(data, X_COLS, Y_COLS, upset, offset, hop) for data in datasets]
    datasets = xqdm(__input_target_split, items, desc='Input target split', ncols=100)

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
    return X_train, Y_train, X_valid, Y_valid


def radian2degree(radian):
    return radian * 180 / np.pi


def simple_error(X_train, X_valid, Y_train, Y_valid):
    diff1 = Y_train[:, :3] - X_train[:, :3]
    diff2 = Y_valid[:, :3] - X_valid[:, :3]
    mae1 = np.mean(np.abs(diff1), axis=0)
    mae2 = np.mean(np.abs(diff2), axis=0)
    mae = np.mean([mae1, mae2], axis=0)
    mae = radian2degree(mae)
    rms = np.sqrt(np.mean(np.square(mae)))
    return mae, rms


def main(args):
    data_dir = Path(args.data_dir)

    if not args.fixed_filenames:
        csv_files = data_dir.glob('motion_data_*.csv')
    else:
        csv_files = [data_dir / f for f in args.fixed_filenames]

    X_train, Y_train, X_valid, Y_valid = make_dataset(csv_files, args.upset, args.offset, args.hop)
    with h5py.File(data_dir / args.output_file_name, 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('Y_train', data=Y_train)
        f.create_dataset('X_valid', data=X_valid)
        f.create_dataset('Y_valid', data=Y_valid)

    print('Calculate MAE, RMS')
    mae, rms = simple_error(X_train, X_valid, Y_train, Y_valid)
    print('MAE:', mae)
    print('RMS:', rms)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='')
    p.add_argument('data_dir', type=str)
    p.add_argument('output_file_name', type=str)
    p.add_argument('--upset', type=int, default=6, help='input의 길이')
    p.add_argument('--offset', type=int, default=6, help='T')
    p.add_argument('--hop', type=int, default=1)
    p.add_argument('--fixed-filenames', type=str, nargs='*')

    args = p.parse_args(sys.argv[1:])
    main(args)
