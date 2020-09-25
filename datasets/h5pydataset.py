import torch
from torch.utils.data import Dataset


class H5pyDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.Y is not None:
            y = torch.tensor(self.Y[idx], dtype=torch.float32)
            return x, y
        else:
            return x
