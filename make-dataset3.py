"""
데이터에 포함되어있는 outlier를 제거하기 위해 wavelet transform(db1)을 적용한 뒤 데이터의 고주파 성분을 blur 한다.
testset에는 적용하지 않고, 비교를 위해 적용하지 않은 데이터도 따로 만들어둔다.

입력 데이터로는 이미 만들어진 interpolation~.csv 를 이용
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from multiprocessing.dummy import Pool
from numbers import Number
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch_burn as tb

# 각 컬럼별 max, min, mean, std 값들 저장
datainfo = None


def getopt(argv):
    p = argparse.ArgumentParser()
    # IO options
    p.add_argument('out_path', type=str)
    # dataset options
    p.add_argument('--T', type=int, default=6)
    p.add_argument('--window_size', type=int, default=6)
    p.add_argument('--hop_length', type=int, default=6)
    p.add_argument('--columns', type=int, default=3)

    args = p.parse_args(argv)

    args.out_path = Path(args.out_path)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    return args


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


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
    for col in xcols:
        csv[col] = (csv[col] - datainfo[col][2]) / datainfo[col][3]

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
    args = getopt(sys.argv[1:])
    tb.pprint_args(args)
    main(args)
