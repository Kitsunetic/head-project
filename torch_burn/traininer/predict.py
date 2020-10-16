from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Predictor:
    def __init__(self,
                 model: nn.Module,
                 batch_size=32,
                 shuffle=False,
                 drop_last=False,
                 cpus=cpu_count(),
                 gpus=torch.cuda.device_count(),
                 verbose=True):
        self.model = model
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.cpus = cpus
        self.gpus = gpus
        self.verbose = verbose

    def forward(self, data):
        x = data
        if self.gpus:
            x = x.cuda()
        preds = self.model(x).detach().cpu()
        return preds

    def on_predict_begin(self):
        pass

    def on_predict_end(self):
        pass

    def predict(self, dataset: Dataset):
        self.on_predict_begin()

        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                        num_workers=self.cpus, drop_last=self.drop_last)

        t = None
        if self.verbose:
            t = tqdm(total=len(dl), ncols=100, desc='Prediction')

        rets = []
        for data in dl:
            ret = self.forward(data)
            ret[ret > 1] = 1
            ret[ret < 0] = 0
            ret *= 255
            ret = ret.type(torch.uint8)
            rets.append(ret)

            if self.verbose:
                t.update()
        ret = torch.cat(rets)

        if self.verbose:
            t.close()

        self.on_predict_end()
        return ret
