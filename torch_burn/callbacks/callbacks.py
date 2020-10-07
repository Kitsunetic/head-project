import math
from pathlib import Path
from typing import AnyStr

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from torch_burn.metrics import Metric


class Callback:
    priority = 100

    def on_epoch_begin(self, is_train: bool, epoch: int):
        pass

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        pass

    def on_batch_begin(self, is_train: bool, epoch: int):
        pass

    def on_batch_end(self, is_train: bool, epoch: int, losses: dict):
        pass


class MetricImprovingCallback(Callback):
    def __init__(self, monitor: Metric):
        self.monitor = monitor

        self.best_metric_value = math.inf
        if self.monitor.mode == 'max':
            self.best_metric_value *= -1

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        if not is_train:
            metric_name, metric_value = self.get_metric_info(logs)
            condition1 = (self.monitor.mode == 'max' and self.best_metric_value < metric_value)
            condition2 = (self.monitor.mode == 'min' and self.best_metric_value > metric_value)
            if condition1 or condition2:
                self.on_metric_improved(is_train, epoch, logs, metric_name, metric_value)
                self.best_metric_value = metric_value
            else:
                self.on_metric_not_improved(is_train, epoch, logs, metric_name, metric_value)

    def get_metric_info(self, logs: dict):
        metric_name = 'val_' + self.monitor.name
        assert metric_name in logs, f'There is no metric value in logs: {metric_name}'
        metric_value = logs[metric_name]
        return metric_name, metric_value

    def on_metric_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        pass

    def on_metric_not_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        pass


class SaveCheckpoint(MetricImprovingCallback):
    def __init__(self, checkpoint_spec: dict,
                 monitor: Metric,
                 save_dir: AnyStr,
                 filepath: AnyStr = 'ckpt-epoch{epoch:04d}-val_loss{val_loss:.4f}.pth',
                 save_best_only=True, verbose=True):
        super(SaveCheckpoint, self).__init__(monitor)

        self.checkpoint_spec = checkpoint_spec
        self.filepath = Path(save_dir) / filepath
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        if not is_train:
            if self.save_best_only:
                super().on_epoch_end(is_train, epoch, logs)
            else:
                metric_name, metric_value = self.get_metric_info(logs)
                self.on_metric_improved(is_train, epoch, logs, metric_name, metric_value)

    def on_metric_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        if not is_train:
            if self.verbose:
                text = 'Save checkpoint: '
                text += metric_name
                text += ' decreased ' if self.monitor.mode == 'min' else ' increased '
                text += f'from {self.best_metric_value} to {metric_value}'
                print(text)
            self.best_metric_value = metric_value

            filepath = str(self.filepath).format(epoch=epoch, **logs)
            data = {}
            for k, v in self.checkpoint_spec.items():
                if isinstance(self.checkpoint_spec[k], nn.DataParallel):
                    data[k] = v.module.state_dict()
                elif isinstance(self.checkpoint_spec[k], nn.Module) or isinstance(self.checkpoint_spec[k], Optimizer):
                    data[k] = v.state_dict()
                else:
                    data[k] = v
            torch.save(data, filepath)


class SaveSampleBase(Callback):
    """
    Save one sample per a epoch
    Must be specified how to save sample data through `save_data` overriding function.
    """

    def __init__(self, model: nn.Module, sample_input: torch.Tensor, save_dir: AnyStr,
                 filepath: AnyStr = 'sample-epoch{epoch:04d}-val_loss{val_loss:.4f}.png',
                 sample_input_filename: AnyStr = None,
                 sample_gt: torch.Tensor = None, sample_gt_filename: AnyStr = None,
                 cuda=True, verbose=True):
        """

        :param model: model which already uploaded on GPU if you are tending to use GPU
        :param sample_input: an input tensor data which could be directly fed into the model
        :param filepath: adaptive filepath string
        :param sample_input_filename: if not None and save_input is overloaded, write sample_input as a file
        :param sample_gt: if sample_gt and sample_gt_filename is not None, write sample ground truth as a file
        :param sample_gt_filename:
        :param verbose:
        """
        self.model = model
        self.sample_input = sample_input
        self.filepath = Path(save_dir) / filepath
        self.verbose = verbose
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.cuda = cuda

        if sample_input_filename:
            fpath = self.filepath.parent / sample_input_filename
            print('Write sample input:', fpath)
            self.save_input(self.sample_input, fpath)

        if sample_gt is not None and sample_gt_filename is not None:
            fpath = self.filepath.parent / sample_gt_filename
            print('Write sample gt:', fpath)
            self.save_gt(sample_gt, fpath)

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        if not is_train:
            with torch.no_grad():
                self.model.eval()

                filepath = str(self.filepath).format(epoch=epoch, **logs)
                x = self.sample_input.unsqueeze(0)
                if self.cuda:
                    x = x.cuda()
                out = self.model(x)

                if self.verbose:
                    print('Write sample', Path(filepath).name)
                self.save_data(out, filepath)

    def save_input(self, input: torch.Tensor, filepath: str):
        """Specify how to save input data"""
        pass

    def save_data(self, output: torch.Tensor, filepath: str):
        """Specify how to save output data"""
        raise NotImplementedError('SaveSampleBase is abstract class which must be inherited')

    def save_gt(self, gt: torch.Tensor, filepath: str):
        """Specify how to save ground truth data"""
        pass


class EarlyStopping(MetricImprovingCallback):
    stopped = False

    def __init__(self, monitor: Metric, patience=10, verbose=True):
        super(EarlyStopping, self).__init__(monitor)

        self.patience = patience
        self.verbose = verbose

        self.stopping_cnt = 0

    def on_metric_not_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        if not is_train:
            self.stopping_cnt += 1
            print('val_loss is not improved for', self.stopping_cnt, 'epochs')
            if self.stopping_cnt >= self.patience:
                if self.verbose:
                    metric_name, metric_value = super().get_metric_info(logs)
                    print('Stop training because', metric_name, 'did not improved for', self.patience, 'epochs')
                self.stopped = True

    def on_metric_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        if not is_train:
            self.stopping_cnt = 0


class LRDecaying(MetricImprovingCallback):
    def __init__(self, optim: Optimizer, monitor: Metric, patience=5, decay_rate=0.5, verbose=True):
        super(LRDecaying, self).__init__(monitor)

        self.optim = optim
        self.patience = patience
        self.decay_rate = decay_rate
        self.verbose = verbose

        # self.lr = next(iter(self.optim.param_groups.values()))['lr']
        self.lr = self.optim.param_groups[0]['lr']

        self.decaying_cnt = 0

    def on_metric_not_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        if not is_train:
            self.decaying_cnt += 1
            if self.decaying_cnt >= self.patience:
                self.decaying_cnt = 0
                new_lr = self.lr * self.decay_rate
                if self.verbose:
                    metric_name, metric_value = self.get_metric_info(logs)
                    print('Decaying lr from', self.lr, 'to', new_lr,
                          'because', metric_name, 'did not improved for', self.patience, 'epochs')

                for p in self.optim.param_groups:
                    p['lr'] = new_lr
                self.lr = new_lr

    def on_metric_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        if not is_train:
            self.decaying_cnt = 0


class Tensorboard(Callback):
    def __init__(self, logdir: AnyStr, model: nn.Module = None,
                 sample_input: torch.Tensor = None, comment: str = '',
                 gpus=torch.cuda.device_count()):
        self.writer = SummaryWriter(logdir, comment=comment)

        if model is not None and sample_input is not None:
            with torch.no_grad():
                model.eval()

                sample_input = sample_input.unsqueeze(0)
                if gpus > 0:
                    sample_input = sample_input.cuda()
                self.writer.add_graph(model, sample_input)

    def on_batch_end(self, is_train: bool, epoch: int, losses: dict):
        for k, v in losses.items():
            self.writer.add_scalar(k, v, epoch)
