import torch
import torch.nn as nn


class Metric:
    def __init__(self, name: str, mode='min', visible=True):
        mode = mode.lower()
        assert mode in ['min', 'max']

        self.name = name
        self.mode = mode
        self.visible = visible

    def on_train_epoch_begin(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_valid_epoch_begin(self):
        pass

    def on_valid_epoch_end(self):
        pass

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        pass


class InvisibleMetric(Metric):
    def __init__(self, name: str, mode='min'):
        super(InvisibleMetric, self).__init__(name=name, mode=mode, visible=False)


class ModuleMetric(Metric):
    def __init__(self, module: nn.Module, name: str, mode='min', visible=True):
        super(ModuleMetric, self).__init__(name, mode)
        self.module = module

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        loss = self.module(outputs, targets)
        return loss


class CSIMetric(InvisibleMetric):
    def __init__(self, name: str, threshold: float = 0.5):
        """
        CSI(Critical Success Index)

        CSI = hits / (hits + misses + falsealarm)
            range: 0~1. Good when near to 1.

        Precision = hits / (hits + falsealarm)
        Recall = hits / (hits + misses)
        F1 Score = 2 * precision * recall / (precision + recall)

        Parameters
        ----------
        name
        threshold
        """
        super(CSIMetric, self).__init__(name=name, mode='max')

        self.threshold = threshold

        self.batches = 0
        self.hits = 0
        self.misses = 0
        self.falsealarm = 0
        self.truenegative = 0
        self.is_valid = False

    def on_train_epoch_begin(self):
        self.is_valid = False

    def on_valid_epoch_begin(self):
        self.batches = 0
        self.hits = 0
        self.misses = 0
        self.falsealarm = 0
        self.truenegative = 0
        self.is_valid = True

    def on_valid_epoch_end(self):
        self.hits = self.hits / self.batches
        self.misses = self.misses / self.batches
        self.falsealarm = self.falsealarm / self.batches
        self.truenegative = self.truenegative / self.batches

        print(' - Hits        :', self.hits)
        print(' - Misses      :', self.misses)
        print(' - FalseAlarm  :', self.falsealarm)
        print(' - TrueNegative:', self.truenegative)

        self.csi = -1
        if not (self.hits == self.misses == self.falsealarm == 0):
            self.csi = self.hits / (self.hits + self.misses + self.falsealarm)
        print(' - CSI         :', self.csi)

        self.precision = -1
        if not (self.hits == self.falsealarm == 0):
            self.precision = self.hits / (self.hits + self.falsealarm)
        print(' - Precision   :', self.precision)

        self.recall = -1
        if not (self.hits == self.misses == 0):
            self.recall = self.hits / (self.hits + self.misses)
        print(' - Recall      :', self.recall)

        self.f1score = -1
        if not (self.precision == self.recall == 0):
            self.f1score = 2 * self.precision * self.recall / (self.precision + self.recall)
        print(' - F1 Score    :', self.f1score)

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        if not self.is_valid:
            return

        with torch.no_grad():
            b = outputs.shape[0]
            x = torch.flatten(outputs)
            y = torch.flatten(targets)
            x[x >= self.threshold] = 1
            y[y < self.threshold] = 0
            dx = 1 - x
            dy = 1 - y

            hits = int(torch.sum(x * y).item())
            misses = int(torch.sum(dx * y).item())
            falsealarm = int(torch.sum(x * dy).item())
            truenegative = int(torch.sum(dx * dy).item())

            self.batches += b
            self.hits += hits
            self.misses += misses
            self.falsealarm += falsealarm
            self.truenegative += truenegative
