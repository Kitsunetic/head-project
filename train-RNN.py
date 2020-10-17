import argparse
import sys
from multiprocessing import cpu_count
from pathlib import Path

import torch
import torch.nn as nn

import datasets
import models
import torch_burn as tb
import utils
from torch_burn.callbacks import EarlyStopping, SaveCheckpoint, LRDecaying, Tensorboard
from torch_burn.metrics import ModuleMetric
from torch_burn.traininer import Trainer2


def getopt(argv):
    p = argparse.ArgumentParser()

    # train options
    p.add_argument('--checkpoint-dir', type=str, default='./checkpoint')
    p.add_argument('--num-epochs', type=int, default=20)
    p.add_argument('--start-epoch', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    # experiment options
    p.add_argument('--cpus', type=int, default=-1)
    p.add_argument('--gpus', type=int, default=-1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('experiment_name', type=str)
    p.add_argument('dataset_path', type=str, help='example: data/head/head-dataset-3166.hdf5')

    args = p.parse_args(argv)
    tb.seed_everything(args.seed)

    if args.cpus < 0:
        args.cpus = cpu_count()
    if args.gpus < 0:
        args.gpus = torch.cuda.device_count()

    args.dataset_path = Path(args.dataset_path)
    args.checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return args


def get_metrics(criterion, datainfo):
    metrics = [ModuleMetric(criterion, 'loss')]

    mean_yaw = datainfo['input_orientation_yaw'][2]
    std_yaw = datainfo['input_orientation_yaw'][3]
    mean_pitch = datainfo['input_orientation_pitch'][2]
    std_pitch = datainfo['input_orientation_pitch'][3]
    mean_roll = datainfo['input_orientation_roll'][2]
    std_roll = datainfo['input_orientation_roll'][3]
    metrics.append(utils.YawMetric('yaw', mean=mean_yaw, std=std_yaw))
    metrics.append(utils.PitchMetric('pitch', mean=mean_pitch, std=std_pitch))
    metrics.append(utils.RollMetric('roll', mean=mean_roll, std=std_roll))
    metrics.append(utils.RMSMetric('rms', yaw_std=std_yaw, pitch_std=std_pitch, roll_std=std_roll))
    return metrics


def main(args):
    train_ds, valid_ds, datainfo = datasets.make_dataset(args.dataset_path)

    model = models.BaselineCNN1d(in_channels, 3)
    criterion = nn.MSELoss()
    if args.gpus > 0:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics = get_metrics(criterion, datainfo)

    callbacks = [EarlyStopping(metrics[0])]
    trainer = Trainer2(model, optim=optimizer, metrics=metrics, callbacks=callbacks,
                       data_parallel=False, ncols=100, cpus=args.cpus, gpus=args.gpus)
    trainer.fit(train_ds, valid_ds,
                start_epoch=args.start_epoch, num_epochs=args.num_epochs,
                batch_size=args.batch_size, shuffle=False)


if __name__ == '__main__':
    args = getopt(sys.argv[1:])
    tb.pprint_args(args)
    main(args)
