import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

import datasets
import models
import utils
from torch_burn.callbacks import EarlyStopping, SaveCheckpoint, LRDecaying, Tensorboard
from torch_burn.metrics import ModuleMetric
from torch_burn.traininer import Trainer


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
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    gpus = torch.cuda.device_count()

    train_ds, valid_ds, datainfo = datasets.make_dataset(args.dataset_path)

    # 데이터셋 파일 이름에서 in_channels 를 구한다.
    if args.dataset_path.name.startswith('C1'):
        in_channels = 3
    elif args.dataset_path.name.startswith('C2'):
        in_channels = 6
    elif args.dataset_path.name.startswith('C3'):
        in_channels = 10
    else:
        raise NotImplementedError(f'Worng dataset name: {args.dataset_path}')

    model = models.BaselineCNN1d(in_channels, 3)
    criterion = nn.MSELoss()
    if gpus > 0:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics = get_metrics(criterion, datainfo)

    callbacks = [EarlyStopping(metrics[0]),
                 SaveCheckpoint(checkpoint_spec={'model': model, 'optim': optimizer},
                                save_dir=checkpoint_dir,
                                filepath='ckpt-{val_loss:.4f}.pth',
                                monitor=metrics[0]),
                 LRDecaying(optimizer, metrics[0]),
                 Tensorboard(checkpoint_dir, comment=args.dataset_path, gpus=gpus)]
    trainer = Trainer(model, optimizer, metrics, callbacks, ncols=150)
    trainer.fit(train_ds, valid_ds,
                start_epoch=args.start_epoch, num_epochs=args.num_epochs,
                batch_size=args.batch_size, shuffle=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # train options
    p.add_argument('--checkpoint-dir', type=str, default='./checkpoint')
    p.add_argument('--num-epochs', type=int, default=20)
    p.add_argument('--start-epoch', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)

    # mandatory options
    p.add_argument('experiment_name', type=str)
    p.add_argument('dataset_path', type=str, help='example: data/head/head-dataset-3166.hdf5')
    args = p.parse_args(sys.argv[1:])
    args.dataset_path = Path(args.dataset_path)
    main(args)
