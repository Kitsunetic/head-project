import math
import uuid
from pathlib import Path
from typing import AnyStr

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

import G
from metrics import BaseMetric


class BaseCallback:
    def __init__(self):
        self._id = uuid.uuid4().hex
        G._I_callback_ids[self._id] = self

    def on_epoch_begin(self, is_train: bool, epoch: int, logs: dict):
        pass

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        pass

    def on_batch_begin(self, is_train: bool, epoch: int, logs: dict, inputs: torch.tensor):
        pass

    def on_batch_end(self, is_train: bool, epoch: int, logs: dict, inputs: torch.tensor, outputs: torch.tensor):
        pass


class SaveCheckpoint(BaseCallback):
    def __init__(self, checkpoint_spec: dict,
                 filepath: AnyStr, monitor: BaseMetric,
                 save_best_only=True, verbose=True):
        """

        :param checkpoint_spec:
            >>> checkpoint_spec = {'model_name': model,
                                   'optim_name': optim,
                                   ...}
        :param filepath:
        :param monitor:
        :param save_best_only:
        :param save_model_only:
        :param verbose:
        """
        super(SaveCheckpoint, self).__init__()

        self.checkpoint_spec = checkpoint_spec
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.last_metric_value = math.inf
        if self.monitor.mode == 'max':
            self.last_metric_value *= -1

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        if not is_train:
            metric_name = 'val_' + self.monitor.name
            assert metric_name in logs, f'There is no metric value in logs: {metric_name}'

            metric_value = logs[metric_name]
            condition1 = (self.monitor.mode == 'max' and self.last_metric_value < metric_value)
            condition2 = (self.monitor.mode == 'min' and self.last_metric_value > metric_value)
            if condition1 or condition2:
                if self.verbose:
                    text = 'Save checkpoint: '
                    text += metric_name
                    text += ' decreased ' if self.monitor.mode == 'min' else ' increased '
                    text += f'from {self.last_metric_value} to {metric_value}'
                    print(text)
                self.last_metric_value = metric_value

                filepath = str(self.filepath).format(epoch=epoch, **logs)
                data = {}
                for k, v in self.checkpoint_spec.items():
                    vtype = type(self.checkpoint_spec[k])
                    if issubclass(vtype, nn.DataParallel):
                        data[k] = v.module.state_dict()
                    elif issubclass(vtype, nn.Module) or issubclass(vtype, Optimizer):
                        data[k] = v.state_dict()
                    else:
                        data[k] = v
                torch.save(data, filepath)


class SaveSampleBase(BaseCallback):
    """
    Save one sample per a epoch
    Must be specified how to save sample data through `save_data` overriding function.
    """

    def __init__(self, model: nn.Module, sample_input: torch.Tensor, filepath: AnyStr, verbose=False):
        super(SaveSampleBase, self).__init__()
        self.model = model
        self.sample_input = sample_input
        self.filepath = Path(filepath)
        self.verbose = verbose

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        if not is_train:
            with torch.no_grad():
                filepath = str(self.filepath).format(epoch=epoch, **logs)
                device = next(self.model.parameters()).device
                x = self.sample_input.to(device)
                out = self.model(x)
                self.save_data(out, filepath)

    def save_data(self, output: torch.tensor, filepath: str):
        raise NotImplementedError


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose

    def __call__(self, history):
        if len(history['valid']['loss']) > self.patience + 1:
            h = history['valid']['loss'][-self.patience:]
            if min(h) == h[0]:
                if self.verbose:
                    print('Stop training because val_loss did not decreased for', self.patience, 'epochs from', h[0])
                return True
        return False


class LRDecaying:
    def __init__(self, optim_encoder, optim_decoder, patience=5, decay_rate=0.5, verbose=False):
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder
        self.patience = patience
        self.decay_rate = decay_rate
        self.verbose = verbose

        self.lr_encoder = next(iter(self.optim_encoder.param_groups.values()))['lr']
        self.lr_decoder = next(iter(self.optim_decoder.param_groups.values()))['lr']

    def __call__(self, history):
        if len(history['valid']['loss']) > self.patience + 1:
            h = history['valid']['loss'][-self.patience:]
            if min(h) == h[0]:
                new_lr_encoder = self.lr_encoder * self.decay_rate
                new_lr_decoder = self.lr_decoder * self.decay_rate
                if self.verbose:
                    print('LR decaying: encoder from', self.lr_encoder, 'to', new_lr_encoder,
                          'decoder from', self.lr_decoder, 'to', new_lr_decoder)

                for p in self.optim_encoder.param_groups:
                    p['lr'] = new_lr_encoder
                for p in self.optim_decoder.param_groups:
                    p['lr'] = new_lr_decoder

                self.lr_encoder = new_lr_encoder
                self.lr_decoder = new_lr_decoder
