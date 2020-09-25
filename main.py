import argparse
import sys
import time

import h5py
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

import G
import datasets
import models
import utils


def display_result(desc: str, history: utils.History, losses, diffs):
    # calculate MAE / RMS
    with torch.no_grad():
        mae = torch.mean(torch.abs(torch.cat(diffs)), dim=0)
        mae = utils.radian2degree(mae)
        rms = torch.sqrt(torch.sum(mae) / 3)

        loss = sum(losses) / len(losses)
        yaw = mae[0].item()
        pitch = mae[1].item()
        roll = mae[2].item()
        rms = rms.item()

        history.loss.append(loss)
        history.yaw.append(yaw)
        history.pitch.append(pitch)
        history.roll.append(roll)
        history.rms.append(rms)

        msg = desc + '\t'
        msg += f'loss {loss:.4f}\t'
        msg += f'yaw {yaw.item():.4f}\t'
        msg += f'pitch {pitch.item():.4f}\t'
        msg += f'roll {roll.item():.4f}\t'
        msg += f'rms: {rms.item():.4f}'
        time.sleep(0.01)


def train(model: nn.Module, criterion: nn.Module, optim: Optimizer, history: utils.History, dl: DataLoader, desc: str):
    model.train()
    losses = []
    diffs = []
    with tqdm(total=len(dl), desc=desc, ncols=100) as t:
        for x, y in dl:
            if G.GPU:
                x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            diffs.append((y - output).detach())
            mean_loss = sum(losses) / len(losses)
            t.set_postfix_str(f'loss {mean_loss:.4f}', refresh=False)
            t.update()
    display_result(desc, history, losses, diffs)


def valid(model: nn.Module, criterion: nn.Module, history: utils.History, dl: DataLoader, desc: str):
    with torch.no_grad():
        losses = []
        diffs = []
        model.eval()
        with tqdm(total=len(dl), desc=desc, ncols=100) as t:
            for x, y in dl:
                if G.GPU:
                    x, y = x.cuda(), y.cuda()
                output = model(x)
                loss = criterion(output, y)

                losses.append(loss.item())
                diffs.append((y - output).detach())
                mean_loss = sum(losses) / len(losses)
                t.set_postfix_str(f'val_loss {mean_loss:.4f}', refresh=False)
                t.update()
        history.loss.append(losses)
        display_result(desc, history, losses, diffs)


def main(args):
    num_epochs = 20
    batch_size = 64
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with h5py.File('data/head/head-dataset-3166.hdf5', 'r') as f:
        X_train = f['X_train'][()]
        X_valid = f['X_valid'][()]
        Y_train = f['Y_train'][()]
        Y_valid = f['Y_valid'][()]

    train_ds = datasets.H5pyDataset(X_train, Y_train)
    valid_ds = datasets.H5pyDataset(X_valid, Y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = models.FullyConnectedModel()
    summary(model, (1, 18))

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model = model.to(device)

    train_history = utils.History()
    valid_history = utils.History()
    for epoch in range(1, num_epochs + 1):
        desc = f'[{epoch:03d}/{num_epochs:03d}] '
        train(model, criterion, optimizer, train_history, train_dl, desc + 'Train')
        valid(model, criterion, valid_history, valid_dl, desc + 'Validation')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('experiment_name', type=str)
    args = p.parse_args(sys.argv[1:])
    main(args)
