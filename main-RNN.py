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
from torch_burn.callbacks import EarlyStopping
from torch_burn.metrics import ModuleMetric
from torch_burn.traininer import Trainer2


def getopt(argv):
    p = argparse.ArgumentParser()

    # train options
    p.add_argument('--checkpoint-dir', type=str, default='./checkpoint')
    p.add_argument('--num-epochs', type=int, default=50)
    p.add_argument('--start-epoch', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    # experiment options
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpus', type=int, default=-1)
    p.add_argument('--gpus', type=int, default=-1)
    p.add_argument('experiment_name', type=str)
    # p.add_argument('dataset_path', type=str, help='example: data/pth/C2-T18-win48-hop24.pth')

    args = p.parse_args(argv)
    # args.dataset_path = Path(args.dataset_path)
    args.dataset_path = 'data/pth_nostd/C2-T18-win48-hop1.pth'
    args.checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.cpus == -1: args.cpus = cpu_count()
    if args.gpus == -1: args.gpus = torch.cuda.device_count()
    tb.seed_everything(args.seed)

    return args


def main(args):
    train_ds, valid_ds, datainfo = datasets.make_dataset(args.dataset_path)

    model = models.RNNBaseline(input_size=6, hidden_size=12, num_layers=3)
    criterion = nn.MSELoss()
    if args.gpus > 0:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    head_metric = utils.HeadProjectMetric('head')
    metrics = [ModuleMetric(criterion, 'loss'), head_metric]

    callbacks = [EarlyStopping(metrics)]
    trainer = Trainer2(model, optimizer, metrics, callbacks, ncols=100)
    trainer.fit(train_ds, valid_ds,
                start_epoch=args.start_epoch, num_epochs=args.num_epochs,
                batch_size=args.batch_size, shuffle=True)

    # metric 결과를 csv 파일로 저장
    csv_filename = args.checkpoint_dir / 'metrics.csv'
    print('Write metric result into', csv_filename)
    head_metric.to_csv(csv_filename)


if __name__ == '__main__':
    args = getopt(sys.argv[1:])
    main(args)
