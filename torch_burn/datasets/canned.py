import shutil
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import AnyStr

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CannedDataset(Dataset):
    def __init__(self, ds: Dataset, can_path: AnyStr = None):
        """
        Canning function serialize each dataset outputs into '*.pth' files and store to disk
        so that omit preprocessing during training to increase GPU utilization.

        On the first run of canned dataset, all each dataset items will be serialized and stored.
        If storage path for serialized files is not specified, all files will be saved in temp directory
        and will be deleted when program finished.

        :param ds: pytorch Dataset class
        :param can_path: directory to save canned items
        :return: Canned dataset
        """
        super(CannedDataset, self).__init__()
        self.ds = ds
        self.can_path = can_path if can_path is not None else tempfile.mkdtemp()

        self.can_files = CannedDataset.can_dataset(self.ds, self.can_path)

    def __del__(self):
        try:
            shutil.rmtree(self.can_path)
        except:
            pass

    @staticmethod
    def _can_dataset(args):
        ds, i, can_path = args
        can_path = can_path / f'can{i:08d}.pth'
        torch.save(ds[i], can_path)
        return can_path.name

    @staticmethod
    def can_dataset(ds: Dataset, can_path: AnyStr, verbose=True):
        can_path = Path(can_path)
        can_path.mkdir(parents=True, exist_ok=True)

        items = [(ds, i, can_path) for i in range(len(ds))]
        with Pool() as pool:
            if verbose:
                t = tqdm(total=len(ds), ncols=100, desc='Canning ' + ds.__class__.__name__)
            filenames = [None for _ in range(len(items))]
            for i, filename in enumerate(pool.imap_unordered(CannedDataset._can_dataset, items)):
                filenames[i] = filename
                if verbose:
                    t.set_postfix_str(filename, refresh=False)
                    t.update()
            if verbose:
                t.close()
        return filenames

    def __len__(self):
        return len(self.can_files)

    def __getitem__(self, idx):
        return torch.load(self.can_files[idx])
