import datasets, argparse, sys
from pathlib import Path
import h5py

def main(args):
    dataset_path = Path(args.dataset_path)



    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='head-project')
    p.add_argument('dataset_path', type=str)
    args = p.parse_args(sys.argv[1:])
    main(args)
