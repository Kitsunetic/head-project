import torch
import torch.nn as nn


class Metric:
    def __init__(self, name: str, mode='min'):
        mode = mode.lower()
        assert mode in ['min', 'max']

        self.name = name
        self.mode = mode

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        pass


class ModuleMetric(Metric):
    def __init__(self, module: nn.Module, name: str, mode='min'):
        super(ModuleMetric, self).__init__(name, mode)
        self.module = module

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        loss = self.module(outputs, targets)
        return loss
