import math
import time
from multiprocessing import cpu_count
from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from torch_burn.callbacks import Callback, EarlyStopping
from torch_burn.metrics import Metric

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optim: Union[Optimizer, Iterable[Optimizer]],
                 metrics: Union[Metric, Iterable[Metric]],
                 callbacks: Union[Callback, Iterable[Callback]] = None,
                 verbose: bool = True,
                 desc: str = '[{epoch:04d}/{num_epochs:04d}]',
                 ncols: int = None,
                 data_parallel: bool = False):
        """

        :param model: a pytorch model
        :param optim: one or more optimizers
        :param metrics: one or more metrics, the first metrics will be loss
        :param callbacks: one or more callbacks
        :param verbose: show progressbar or not
        :param desc: progressbar description with parameters 'epoch', 'num_epochs'
        :param ncols: width of the progressbar
        :param data_parallel: use data_parallel
        """
        self.model = model
        self.optim = list(optim) if isinstance(optim, Iterable) else [optim]
        self.metrics = list(metrics) if isinstance(metrics, Iterable) else [metrics]
        self.callbacks = list(callbacks) if isinstance(callbacks, Iterable) else [callbacks]
        self.verbose = verbose
        self.desc = desc
        self.ncols = ncols
        self.data_parallel = data_parallel

        self.metrics[0].name = 'loss'

        # set devices
        self._device = next(iter(model.parameters()))[0].device

        # data parallel
        if self.data_parallel:
            self.model = nn.DataParallel(model)

    def forward(self, x, y):
        return self.model(x), y

    def _init_logs(self):
        logs = {metric.name: (math.inf if metric.mode == 'min' else -math.inf) for metric in self.metrics}
        logs.update({'val_' + k: v for k, v in logs.items()})
        return logs

    def fit(self,
            train_dataset: Dataset,
            valid_dataset: Dataset = None,
            train_valid_split: float = None,
            num_epochs: int = 1,
            start_epoch: int = 1,
            batch_size=32,
            shuffle=True,
            num_workers=cpu_count(),
            drop_last=False):
        """
        Train and validate model with specified dataset

        Parameters
        ----------
        train_dataset :
        valid_dataset :
        train_valid_split :
        num_epochs :
        start_epoch :
        batch_size :
        shuffle :
        num_workers :
        drop_last :

        Returns
        -------

        """
        # make dataset
        assert valid_dataset is None or train_valid_split is None

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        if valid_dataset is None and train_valid_split is not None:
            assert 0 < train_valid_split < 1

            dataset_size = len(train_dataset)
            valid_size = int(dataset_size * train_valid_split)
            train_size = dataset_size - valid_size
            train_dataset, valid_dataset = random_split(train_dataset, (train_size, valid_size))

        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, drop_last=drop_last)
        if valid_dataset is not None:
            valid_dl = DataLoader(valid_dataset, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=drop_last)
        else:
            valid_dl = None

        logs = {metric.name: math.inf if metric.mode == 'min' else -math.inf for metric in self.metrics}
        if valid_dl is not None:
            logs.update(
                {'val_' + metric.name: math.inf if metric.mode == 'min' else -math.inf for metric in self.metrics})

        loop_stopper = False
        for epoch in range(start_epoch, num_epochs+1):
            if loop_stopper: # early stopping
                break

            desc_base = self.desc.format(epoch=epoch, num_epochs=num_epochs)
            logs = self._init_logs()

            # train
            for callback in self.callbacks:
                callback.on_epoch_begin(True, epoch)
            self.model.train()
            self.loop(True, epoch, train_dl, logs, tqdm_desc=desc_base + ' Train')
            for callback in self.callbacks:
                callback.on_epoch_end(True, epoch, logs)

            # validation
            if valid_dl is not None:
                for callback in self.callbacks:
                    callback.on_epoch_begin(False, epoch)
                with torch.no_grad():
                    self.model.eval()
                    self.loop(False, epoch, valid_dl, logs, tqdm_desc=desc_base + ' Validation')
                for callback in self.callbacks:
                    callback.on_epoch_end(False, epoch, logs)

                    # early stopping
                    if isinstance(callback, EarlyStopping):
                        if callback.stopped:
                            loop_stopper = True
                            break

    def loop(self, is_train: bool, epoch: int, dl: DataLoader, logs, tqdm_desc: str = ''):
        """
        Each loop for training and validation

        Parameters
        ----------
        is_train :
        epoch :
        dl :
        logs :
        tqdm_desc :

        Returns
        -------

        """
        losses = {('' if is_train else 'val_') + metric.name: 0 for metric in self.metrics}

        with tqdm(total=len(dl), ncols=self.ncols, desc=tqdm_desc) as t:
            for i, data in enumerate(dl):
                for callback in self.callbacks:
                    callback.on_batch_begin(is_train, epoch)

                if not isinstance(data, Iterable) or isinstance(data, torch.Tensor):
                    data = (data,)

                # match device with model
                new_data = []
                for d in data:
                    if isinstance(d, torch.Tensor):
                        new_data.append(d.to(self._device))
                    else:
                        new_data.append(d)
                data = new_data

                # forward
                out = self.forward(*data)
                assert isinstance(out, Tuple) and len(out) >= 2, \
                    f'output of forward must be Tuple whose length larger than 2'

                # backward
                metric = self.metrics[0]
                loss = metric.get_value(*out[:2])
                if is_train:
                    for optim in self.optim:
                        optim.zero_grad()
                    loss.backward()
                    for optim in self.optim:
                        optim.step()

                # calculate metrics
                name = ('val_' if not is_train else '') + 'loss'
                losses[name] = loss.item()
                logs[name] = _ignition_mean(logs[name], loss.item(), i)

                pred, target = out[0].detach(), out[1].detach()
                with torch.no_grad():
                    self.model.eval()
                    for metric in self.metrics[1:]:
                        name = ('val_' if not is_train else '') + metric.name
                        value = metric.get_value(pred, target)
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        losses[name] = value
                        logs[name] = _ignition_mean(logs[name], value, i)

                # update progressbar
                msgs = []
                for metric in self.metrics:
                    name = ('val_' if not is_train else '') + metric.name
                    msg = f'{name} {logs[name]:.4f}'
                    msgs.append(msg)
                msg = ' '.join(msgs)
                t.set_postfix_str(msg, refresh=False)
                t.update()

                for callback in self.callbacks:
                    callback.on_batch_end(is_train, epoch, losses)

        time.sleep(0.001)  # for tqdm timing problem


def _ignition_mean(a, b, i):
    if math.isinf(a):
        return b
    else:
        return (a * i + b) / (i + 1)
