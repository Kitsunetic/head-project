from typing import Tuple

from torch.utils.data import Dataset, Subset


def kfold(ds: Dataset, k: int, fold: int) -> Tuple[Dataset, Dataset]:
    """
    데이터셋을 k개로 잘라서 많은 쪽을 train, 작은 쪽을 test 데이터셋으로 분할합니다.
    Parameters
    ----------
    ds : 입력 데이터셋
    k : 폴드의 개수
    fold : 몇 번째 폴드인지

    Returns
    -------
    나눠진 두 개의 train, test 데이터셋
    """
    assert 0 <= fold < k
    assert k > 1
    assert len(ds) >= k

    idx1, idx2 = [], []
    for i in range(len(ds)):
        if i % k == fold:
            idx2.append(i)
        else:
            idx1.append(i)

    return Subset(ds, idx1), Subset(ds, idx2)


class ChainDataset(Dataset):
    def __init__(self, *ds_list: Dataset):
        """
        데이터셋 여러개를 하나의 데이터셋으로 합칩니다.
        데이터셋 순서는 ds_list의 순서로 이뤄집니다.
        Parameters
        ----------
        ds_list :
        """
        self.ds_list = ds_list
        self.len_list = [len(ds) for ds in self.ds_list]
        self.total_len = sum(self.len_list)

        self.idx_list = []
        for i, l in enumerate(self.len_list):
            self.idx_list.extend([i for _ in range(l)])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        didx = self.idx_list[idx]
        return self.ds_list[didx][idx]
