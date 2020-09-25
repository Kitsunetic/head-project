import argparse
import sys
import time
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

import G
import datasets
import models
import utils


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module = None, optim: Optimizer = None,
                 tensorboard: SummaryWriter = None):
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.tensorboard = tensorboard
        self.train_history = utils.History(self.tensorboard)
        self.valid_history = utils.History(self.tensorboard)

    def train(self, dl: DataLoader, epoch: int, desc: str = None):
        self.model.train()

        losses = []
        diffs = []

        with tqdm(total=len(dl), desc=desc, ncols=100) as t:
            for i, (x, y) in enumerate(dl):
                if G.GPU:
                    x, y = x.cuda(), y.cuda()
                output = self.model(x)
                loss = self.criterion(output, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.tensorboard.add_scalar('Loss/train', loss, epoch)
                self.tensorboard.add_scalar('BatchIndex/train', i, epoch)
                losses.append(loss.item())
                diffs.append((y - output).detach())
                mean_loss = sum(losses[-100:]) / len(losses[-100:])
                t.set_postfix_str(f'loss {mean_loss:.4f}', refresh=False)
                t.update()

    def valid(self, dl: DataLoader, epoch: int, desc: str = None):
        with torch.no_grad():
            self.model.eval()

            losses = []
            diffs = []

            with tqdm(total=len(dl), desc=desc, ncols=100) as t:
                for i, (x, y) in enumerate(dl):
                    if G.GPU:
                        x, y = x.cuda(), y.cuda()
                    output = self.model(x)
                    loss = self.criterion(output, y)

                    self.tensorboard.add_scalar('Loss/valid', loss, epoch)
                    self.tensorboard.add_scalar('BatchIndex/valid', i, epoch)
                    losses.append(loss.item())
                    diffs.append((y - output).detach())
                    mean_loss = sum(losses[-100:]) / len(losses[-100:])
                    t.set_postfix_str(f'val_loss {mean_loss:.4f}', refresh=False)
                    t.update()
            display_result(desc, losses, diffs)

    def predict(self, desc: str = None):
        pass


def display_result(desc: str, losses, diffs):
    # calculate MAE / RMS
    with torch.no_grad():
        mae = torch.mean(torch.abs(torch.cat(diffs)), dim=0)
        mae = utils.radian2degree(mae)
        rms = torch.sqrt(torch.sum(mae) / 3)

        loss = sum(losses[:-100]) / len(losses[-100:])
        yaw = mae[0].item()
        pitch = mae[1].item()
        roll = mae[2].item()
        rms = rms.item()

        msg = desc + '\t'
        msg += f'loss {loss:.4f}\t'
        msg += f'yaw {yaw:.4f}\t'
        msg += f'pitch {pitch:.4f}\t'
        msg += f'roll {roll:.4f}\t'
        msg += f'rms: {rms:.4f}'
        time.sleep(0.01)


def train(model: nn.Module, criterion: nn.Module, optim: Optimizer, dl: DataLoader,
          history: utils.HistoryPack, desc: str):
    model.train()

    losses = []
    diffs = []

    with tqdm(total=len(dl), desc=desc, ncols=100) as t:
        for i, (x, y) in enumerate(dl):
            if G.GPU:
                x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            history.writer.add_scalar('Loss/train', loss, history.epoch)
            history.writer.add_scalar('BatchIndex', i, history.epoch)
            losses.append(loss.item())
            diffs.append((y - output).detach())
            mean_loss = sum(losses[-100:]) / len(losses[-100:])
            t.set_postfix_str(f'loss {mean_loss:.4f}', refresh=False)
            t.update()
    display_result(desc, losses, diffs)


def valid(model: nn.Module, criterion: nn.Module, dl: DataLoader,
          epoch: int, writer: SummaryWriter, desc: str):
    with torch.no_grad():
        model.eval()

        losses = []
        diffs = []

        with tqdm(total=len(dl), desc=desc, ncols=100) as t:
            for i, (x, y) in enumerate(dl):
                if G.GPU:
                    x, y = x.cuda(), y.cuda()
                output = model(x)
                loss = criterion(output, y)

                writer.add_scalar('Loss/valid', loss, epoch)
                writer.add_scalar('BatchIndex', i, epoch)
                losses.append(loss.item())
                diffs.append((y - output).detach())
                mean_loss = sum(losses[-100:]) / len(losses[-100:])
                t.set_postfix_str(f'val_loss {mean_loss:.4f}', refresh=False)
                t.update()
        display_result(desc, losses, diffs)


def main(args):
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 64
    lr = 1e-3

    # tensorboard
    writer = SummaryWriter(log_dir=checkpoint_dir / 'log')
    history = utils.HistoryPack(writer, args.start_epoch, desc)

    with h5py.File(args.dataset_path, 'r') as f:
        X_train = f['X_train'][()]
        X_valid = f['X_valid'][()]
        Y_train = f['Y_train'][()]
        Y_valid = f['Y_valid'][()]

    train_ds = datasets.H5pyDataset(X_train, Y_train)
    valid_ds = datasets.H5pyDataset(X_valid, Y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = models.FullyConnectedModel(args.input_size, args.output_size)
    summary(model, (1, args.input_size))  # print model structure
    history.writer.add_graph(model, input_to_model=train_ds[0][0])

    criterion = nn.MSELoss()
    if G.GPU:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        desc = f'[{epoch:03d}/{args.num_epochs:03d}]'
        train(model, criterion, optimizer, train_dl, history, desc)
        valid(model, criterion, valid_dl, history, desc)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir', type=str, default='./checkpoint')
    p.add_argument('--num-epochs', type=int, default=20)
    p.add_argument('--start-epoch', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--input-size', type=int, default=18)
    p.add_argument('--output-size', type=int, default=3)
    p.add_argument('experiment_name', type=str)
    p.add_argument('dataset_path', type=str, help='example: data/head/head-dataset-3166.hdf5')
    args = p.parse_args(sys.argv[1:])
    main(args)
