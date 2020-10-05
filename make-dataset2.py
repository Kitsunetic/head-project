"""
이미 crop, interpolation된 csv파일을 각각 x/y로 나눠서 pth 파일로 만든다.
hopping을 너무 짧게 하면 기존 데이터를 외워서 과적합이 발생하지 않을지?
파일 이름을 input3-hop6-,....pth 이런식으로 짓자
"""
import argparse, sys
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


def main(args):
    files = Path('data/interpolation').glob('interpolation_*.csv')

    data = {''}


    for file in files:
        pass
    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out_path', type=str)
    args = p.parse_args(sys.argv[1:])

    args.out_path = Path(args.out_path)
    args.out_path.parent.mkdir(parents=True,exist_ok=True)
    main(args)