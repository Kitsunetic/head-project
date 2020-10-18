"""
이미 crop, interpolation된 csv파일을 각각 x/y로 나눠서 pth 파일로 만든다.
hopping을 너무 짧게 하면 기존 데이터를 외워서 과적합이 발생하지 않을지?
파일 이름을 input3-hop6-,....pth 이런식으로 짓자
"""
import argparse
import json
import sys
from collections import defaultdict
from multiprocessing.dummy import Pool
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import torch_burn as tb

# 각 컬럼별 max, min, mean, std 값들 저장
datainfo = None


def process_file(file: Path, T: int, window_size: int, hop_length: int, columns: int):
    global datainfo

    csv = pd.read_csv(file)
    w, h = window_size, hop_length

    """
    xcols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
             'angular_vec_x', 'angular_vec_y', 'angular_vec_z',
             'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w',
             'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll']
     """
    if columns == 1:
        xcols = ['input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll']
    elif columns == 2:
        xcols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                 'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll']
    elif columns == 3:
        xcols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                 'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w',
                 'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll']
    else:
        raise NotImplementedError(f'Unknown columns: {columns}')
    ycols = ['input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll']

    # 각 컬럼별 max, min, mean, std 값들
    if datainfo is None:
        with open('config/datainfo.json', 'r') as f:
            datainfo = json.load(f)

    # csv standardization
    # 데이터에 outlier가 있어서 min/max값을 신뢰할 수 없다보니 normalize는 의미가 없다고 생각됨
    """
    for col in xcols:
        csv[col] = (csv[col] - datainfo[col][2]) / datainfo[col][3]
    """

    # csv의 index를 없애기 위해서 list로 변경
    L = len(csv)
    csv = {col: csv[col].to_list() for col in csv.columns}

    data = {'X_train': [], 'Y_train': [], 'X_test': [], 'Y_test': []}
    for i in range(0, L - w - T, h):
        # train과 test의 중간지점은 무시: 데이터를 학습하는게 아니라 외울 수 있음.
        if L // 2 - w <= i <= L // 2 + w:
            continue

        # 데이터의 구조는
        # X: (column 수, window_size)
        # Y: (column 수, 1) --> 그냥 (columnt 수) 의 vector로 하자
        x = [torch.tensor(csv[col][i:i + w], dtype=torch.float32) for col in xcols]
        y = [csv[col][i + w + T] for col in ycols]
        x = torch.stack(x)
        y = torch.tensor(y, dtype=torch.float32)

        # csv의 중간 인덱스를 기준으로 dict에 X_train, X_test, Y_train, Y_test로 나눠서 저장
        if i < L // 2:
            data['X_train'].append(x)
            data['Y_train'].append(y)
        else:
            data['X_test'].append(x)
            data['Y_test'].append(y)

    """ 데이터 이상치 확인
    X_train = torch.stack(data['X_train'])
    dd = []
    for j in range(X_train.shape[0]):
        dd.append(X_train[j, :, 1:] - X_train[j, :, :-1])
    dd = torch.stack(dd)
    for i in range(12):
        q = dd[:, i, :]
        if q.max() > 0.5 or q.min() < -0.5:
            print(i, q.max(), q.min(), q.mean(), q.std())
    """

    return data


def _process_file(args):
    """threading을 위한 process_file의 dummy"""
    return process_file(*args)


def main(args):
    files = list(Path('data/interpolation').glob('interpolation_*.csv'))
    items = [(f, args.T, args.window_size, args.hop_length, args.columns) for f in files]
    total_data = defaultdict(list)  # X_train, X_test, Y_train, Y_test
    with Pool() as pool:
        with tqdm(total=len(items), ncols=100, desc='Making dataset') as t:
            for i, data in enumerate(pool.imap_unordered(_process_file, items)):
                # 데이터를 하나로 합치기
                for key in data.keys():
                    total_data[key].extend(data[key])
                t.set_postfix_str(files[i].name, refresh=False)
                t.update()

    # datainfo도 추가
    total_data['datainfo'] = datainfo

    # 전체 데이터를 pth파일로 저장
    torch.save(total_data, args.out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out_path', type=str)
    p.add_argument('--T', type=int, default=6)
    p.add_argument('--window_size', type=int, default=6)
    p.add_argument('--hop_length', type=int, default=6)
    p.add_argument('--columns', type=int, default=3)
    args = p.parse_args(sys.argv[1:])

    args.out_path = Path(args.out_path)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    tb.pprint_args(args)
    main(args)
