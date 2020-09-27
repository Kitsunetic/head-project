import argparse
import sys
from pathlib import Path

import h5py
import torch
import torch.nn as nn

import datasets
import models
import torch_burn as tb


class YawMetric(tb.metrics.Metric):
    def __call__(self, outputs: torch.tensor, targets: torch.tensor):
        return torch.mean(torch.abs(outputs[:, 0] - targets[:, 0]))


class RollMetric(tb.metrics.Metric):
    def __call__(self, outputs: torch.tensor, targets: torch.tensor):
        return torch.mean(torch.abs(outputs[:, 1] - targets[:, 1]))


class PitchMetric(tb.metrics.Metric):
    def __call__(self, outputs: torch.tensor, targets: torch.tensor):
        return torch.mean(torch.abs(outputs[:, 2] - targets[:, 2]))


class RMSMetric(tb.metrics.Metric):
    def __call__(self, outputs: torch.tensor, targets: torch.tensor):
        return torch.mean(torch.abs(outputs - targets))


def main(args):
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    gpus = torch.cuda.device_count()

    with h5py.File(args.dataset_path, 'r') as f:
        X_train = f['X_train'][()]
        X_valid = f['X_valid'][()]
        Y_train = f['Y_train'][()]
        Y_valid = f['Y_valid'][()]
    train_ds = datasets.H5pyDataset(X_train, Y_train)
    valid_ds = datasets.H5pyDataset(X_valid, Y_valid)

    # model = models.FullyConnectedModel(args.input_size, args.output_size)
    # model = models.BaselineFC2(args.input_size, args.output_size)
    model = models.BaselineFC3(args.input_size, args.output_size)
    criterion = nn.MSELoss()
    if gpus > 0:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metric = tb.metrics.ModuleMetric(criterion, 'loss')
    metrics = [metric,
               YawMetric('yaw'),
               PitchMetric('pitch'),
               RollMetric('roll'),
               RMSMetric('rms')]
    callbacks = [
        tb.callbacks.EarlyStopping(metric),
        tb.callbacks.SaveCheckpoint(checkpoint_spec={'model': model, 'optim': optimizer},
                                    filepath=checkpoint_dir / 'ckpt-{val_loss:.4f}.pth',
                                    monitor=metric),
        tb.callbacks.LRDecaying(optimizer, metric),
        tb.callbacks.Tensorboard(checkpoint_dir, model, train_ds[0][0].cuda(), args.dataset_path)
    ]
    trainer = tb.traininer.Trainer(model, optimizer, metrics, callbacks)
    trainer.fit(train_ds, valid_ds,
                start_epoch=args.start_epoch, num_epochs=args.num_epochs,
                batch_size=args.batch_size, shuffle=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir', type=str, default='./checkpoint')
    p.add_argument('--num-epochs', type=int, default=20)
    p.add_argument('--start-epoch', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--input-size', type=int, default=18)
    p.add_argument('--output-size', type=int, default=3)
    p.add_argument('experiment_name', type=str)
    p.add_argument('dataset_path', type=str, help='example: data/head/head-dataset-3166.hdf5')
    args = p.parse_args(sys.argv[1:])
    main(args)
