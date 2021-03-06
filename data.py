import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CSVSequentialDataset(Dataset):
    def __init__(self, csvfile, window_size, stride):
        super(CSVSequentialDataset, self).__init__()

        self.xcols = ['input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll',
                      'acceleration_x', 'acceleration_y', 'acceleration_z']
        self.ycols = ['input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll']

        self.means = torch.tensor([-2.5188, 7.4404, 0.0633, 0.2250, 9.5808, -1.0252], dtype=torch.float32)
        self.stds = torch.tensor([644.7101, 80.9247, 11.4308, 0.4956, 0.0784, 2.3869], dtype=torch.float32)

        self.csv = pd.read_csv(csvfile)
        for i, col in enumerate(self.xcols):
            self.csv[col] = (self.csv[col] - self.means[i]) / self.stds[i]
        self.window_size = window_size

        self.indexes = []
        i = 0
        while i <= len(self.csv) - window_size * 2:
            self.indexes.append((i, i + window_size))
            i += stride

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        sx, tx = self.indexes[idx]
        ty = tx + 18
        x = self.csv.iloc[sx:tx][self.xcols].to_numpy()
        y = self.csv.iloc[ty][self.ycols]

        x = torch.tensor(x, dtype=torch.float32)  # 300, 6
        y = torch.tensor(y, dtype=torch.float32)  # 1, 3

        return x, y


class SingleFileDataset(Dataset):
    def __init__(self, dataset_file):
        super(SingleFileDataset, self).__init__()

        data = np.load(dataset_file)
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.Y = torch.tensor(data['Y'], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # 300, 6
        y = self.Y[idx]  # 1, 3

        return x, y
