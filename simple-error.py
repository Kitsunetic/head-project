import argparse
import sys
import torch
import numpy as np

import datasets


def main(args):
    train_ds, valid_ds = datasets.InterpolationDataset.make_dataset(args.dataset_path)

    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='head-project Simple Error Test')
    p.add_argument('dataset_path', type=str)
    args = p.parse_args(sys.argv[1:])
    main(args)
