from typing import AnyStr

import h5py
import numpy as np
import torch.utils.data
import torch


class InterpolationDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, is_train: bool):
        super(InterpolationDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.is_train = is_train

    def __len__(self):
        return len(self.X.shape[0])

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.Y[idx])
        return x, y

    @staticmethod
    def make_dataset(dataset_file: AnyStr):
        with h5py.File(dataset_file, 'r') as f:
            X_train: np.ndarray = f['X_train'][()]
            Y_train: np.ndarray = f['Y_train'][()]
            X_valid: np.ndarray = f['X_valid'][()]
            Y_valid: np.ndarray = f['Y_valid'][()]

        train_ds = InterpolationDataset(X_train, Y_train, is_train=True)
        valid_ds = InterpolationDataset(X_valid, Y_valid, is_train=False)
        return train_ds, valid_ds
