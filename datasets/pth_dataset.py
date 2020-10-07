import torch
from torch.utils.data import Dataset


class PTHDataset(Dataset):
    def __init__(self, data, target):
        self.data = data  # torch.Size([13, window_size])
        self.target = target  # torch.Size([3])

        assert len(self.data) == len(self.target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.target[idx]
        return x, y


def make_dataset(filepath):
    data = torch.load(filepath)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    ds_train = PTHDataset(X_train, Y_train)
    ds_test = PTHDataset(X_test, Y_test)
    datainfo = data['datainfo']
    return ds_train, ds_test, datainfo
