import math
import time
from multiprocessing import cpu_count
from typing import List, Union, Iterable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from torch_burn.callbacks import Callback
from torch_burn.datasets.utils import kfold
from torch_burn.metrics import Metric


class Trainer2:
    def __init__(self,
                 model: nn.Module,
                 optim: Union[Optimizer, Iterable[Optimizer]],
                 metrics: Union[Metric, Iterable[Metric]],
                 callbacks: Union[Callback, Iterable[Callback]] = None,
                 desc: str = '[{epoch:04d}/{num_epochs:04d}]',
                 data_parallel: bool = False,
                 gpus: int = torch.cuda.device_count(),
                 cpus: int = cpu_count(),
                 ncols: int = 128,
                 verbose: bool = True):
        self.model = model
        self.optim = list(optim) if isinstance(optim, Iterable) else [optim]
        self.metrics = list(metrics) if isinstance(metrics, Iterable) else [metrics]
        self.callbacks = list(callbacks) if isinstance(callbacks, Iterable) else [callbacks]
        self.desc = desc
        self.data_parallel = data_parallel
        self.gpus = gpus
        self.cpus = cpus
        self.verbose = verbose
        self.ncols = ncols

        self.metrics[0].name = 'loss'

        # sort callbacks along whose priorities
        self.callbacks: List[Callback] = sorted(self.callbacks, key=lambda cb: cb.priority, reverse=True)

        # data parallel
        if self.data_parallel:
            self.model = nn.DataParallel(model)

    def fit(self,
            train_dataset: Dataset,
            valid_dataset: Dataset = None,
            train_valid_split: float = None,
            num_folds: int = None,
            fold: int = None,
            num_epochs: int = 1,
            start_epoch: int = 1,
            batch_size=32,
            shuffle=True,
            drop_last=False):
        train_ds, valid_ds = self._init_dataset(train_dataset, valid_dataset, train_valid_split, num_folds, fold)
        train_dl, valid_dl = self._init_dataloader(train_ds, valid_ds, batch_size, shuffle, drop_last)

        # logs - average metric value of each epochs
        # losses - metric value of each batches
        losses = self._init_train_losses()
        self.stop_loop = False
        for epoch in range(start_epoch, num_epochs + 1):
            if self.stop_loop: break
            logs = self._init_logs()

            # train callbacks
            for cb in self.callbacks:
                cb.on_train_epoch_begin(epoch)
            for m in self.metrics:
                m.on_train_epoch_begin()

            # train loop
            self.model.train()
            with tqdm(total=len(train_dl), ncols=self.ncols,
                      desc=self.desc.format(epoch=epoch, num_epochs=num_epochs) + ' Train') as t:
                for batch_idx, data in enumerate(train_dl):
                    # train batch callbacks
                    for cb in self.callbacks:
                        cb.on_train_batch_begin(epoch, batch_idx)

                    # forward / backward
                    pred, y = self.forward(data)
                    loss = self.metrics[0].get_value(pred, y)
                    for optim in self.optim:
                        optim.zero_grad()
                    loss.backward()
                    for optim in self.optim:
                        optim.step()

                    # metrics
                    losses['loss'] = loss.item()
                    logs['loss'] = _ignition_mean(logs['loss'], loss.item(), batch_idx)

                    pred, y = pred.detach(), y.detach()
                    with torch.no_grad():
                        self.model.eval()
                        for m in self.metrics[1:]:
                            v = m.get_value(pred, y)
                            if m.visible:
                                if isinstance(v, torch.Tensor):
                                    v = v.item()
                                losses[m.name] = v
                                logs[m.name] = _ignition_mean(logs[m.name], v, batch_idx)

                    # update progressbar
                    msgs = []
                    for m in self.metrics:
                        if m.visible:
                            msgs.append(f'{m.name} {logs[m.name]:.4f}')
                    msg = ' '.join(msgs)
                    t.set_postfix_str(msg, refresh=False)
                    t.update()

                    # train batch callbacks
                    for cb in self.callbacks:
                        cb.on_train_batch_end(epoch, batch_idx, losses)

            # wait for tqdm closed
            time.sleep(0.01)

            # train epoch callbacks
            for cb in self.callbacks:
                cb.on_train_epoch_end(epoch, logs)
            for m in self.metrics:
                m.on_train_epoch_end()

            # valid callbacks
            for cb in self.callbacks:
                cb.on_valid_epoch_begin(epoch)
            for m in self.metrics:
                m.on_valid_epoch_begin()

            # valid loop
            with torch.no_grad():
                self.model.eval()
                with tqdm(total=len(valid_dl), ncols=self.ncols,
                          desc=self.desc.format(epoch=epoch, num_epochs=num_epochs) + ' Validation') as t:
                    for batch_idx, data in enumerate(valid_dl):
                        # train batch callbacks
                        for cb in self.callbacks:
                            cb.on_valid_batch_begin(epoch, batch_idx)

                        # forward / backward
                        pred, y = self.forward(data)
                        loss = self.metrics[0].get_value(pred, y)

                        # metrics
                        losses['val_loss'] = loss.item()
                        logs['val_loss'] = _ignition_mean(logs['val_loss'], loss.item(), batch_idx)

                        pred, y = pred.detach(), y.detach()
                        with torch.no_grad():
                            self.model.eval()
                            for m in self.metrics[1:]:
                                v = m.get_value(pred, y)
                                if m.visible:
                                    if isinstance(v, torch.Tensor):
                                        v = v.item()
                                    name = 'val_' + m.name
                                    losses[name] = v
                                    logs[name] = _ignition_mean(logs[name], v, batch_idx)

                        # update progressbar
                        msgs = []
                        for m in self.metrics:
                            if m.visible:
                                name = 'val_' + m.name
                                msgs.append(f'{name} {logs[name]:.4f}')
                        msg = ' '.join(msgs)
                        t.set_postfix_str(msg, refresh=False)
                        t.update()

                        # train batch callbacks
                        for cb in self.callbacks:
                            cb.on_valid_batch_end(epoch, batch_idx, losses)

                # Wait for tqdm closed
                time.sleep(0.01)

                # Validation epoch callbacks
                for cb in self.callbacks:
                    cb.on_valid_epoch_end(epoch, logs)
                for m in self.metrics:
                    m.on_valid_epoch_end()

    def _init_dataset(self,
                      train_dataset: Dataset,
                      valid_dataset: Dataset = None,
                      train_valid_split: float = None,
                      num_folds: int = None,
                      fold: int = None):
        if valid_dataset is not None:
            return train_dataset, valid_dataset
        elif (train_valid_split is not None) and (0 < train_valid_split < 1):
            # train-valid split
            s = len(train_dataset)
            v = int(s * train_valid_split)
            t = s - v
            return random_split(train_dataset, (t, v))
        elif num_folds is not None and fold is not None:
            # k-fold
            return kfold(train_dataset, num_folds, fold)
        elif num_folds is not None or fold is not None:
            raise NotImplementedError('Both num_folds and fold must be specified')
        else:
            # no validation
            return train_dataset, None

    def _init_dataloader(self,
                         train_dataset: Dataset,
                         valid_dataset: Dataset = None,
                         batch_size=32,
                         shuffle=True,
                         drop_last=False):
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=self.cpus, drop_last=drop_last)
        valid_dl = None
        if valid_dataset is not None:
            valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=self.cpus, drop_last=drop_last)

        return train_dl, valid_dl

    def _init_logs(self):
        logs = {}
        for m in self.metrics:
            if m.visible:
                logs[m.name] = math.inf if m.mode == 'min' else -math.inf
                logs['val_' + m.name] = math.inf if m.mode == 'min' else -math.inf
        return logs

    def _init_train_losses(self):
        losses = {}
        for m in self.metrics:
            if m.visible:
                losses[m.name] = 0
        return losses

    def _init_valid_losses(self):
        losses = {}
        for m in self.metrics:
            if m.visible:
                losses['val_' + m.name] = 0
        return losses

    def forward(self, data):
        x, y = data[:2]
        if self.gpus > 0:
            x = x.cuda()
            y = y.cuda()
        return self.model(x), y


def _ignition_mean(a, b, i):
    if math.isinf(a):
        return b
    else:
        return (a * i + b) / (i + 1)
