from pathlib import Path
import argparse
import sys
import os
import numpy as np
import torch


def main(args):
    pass


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--networks', type=str, required=True)
    argparser.add_argument('--dataset', type=str, required=True)
    argparser.add_argument('--result', type=str, default='results')

    args = argparser.parse_args(sys.argv[1:])
    args.result = Path(args.result)
