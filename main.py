import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_optimizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models


def main(args):
    checkpoint_dir = Path(args.checkpoint_dir, args.experiment_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    train_ds, valid_ds = datasets.InterpolationDataset.make_dataset(args.dataset_path)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    print(f'Dataset: train[{len(train_ds)}] validation[{len(valid_ds)}]')

    # load model
    model = models.from_name(args.model_name)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    #optimizer = torch_optimizer.RAdam(model.parameters(), lr=args.lr)
    optimizer = torch_optimizer.SGD(model.parameters(), lr=args.lr)

    min_loss = math.inf
    early_stopping_cnt = 0
    lr_decay_cnt = 0
    for epoch in range(1, args.num_epochs + 1):
        # train
        model.train()
        losses = []
        with tqdm(total=len(train_ds), ncols=100, desc=f'[{epoch:03d}/{args.num_epochs:03d}] Train') as t:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update progress
                losses.append(loss.item())
                mean_loss = sum(losses) / len(losses)
                t.set_postfix_str(f'loss: {mean_loss:.4f}', refresh=False)
                t.update(n=args.batch_size)

        # validation
        with torch.no_grad():
            model.eval()
            losses = []
            with tqdm(total=len(train_ds), ncols=100, desc=f'[{epoch:03d}/{args.num_epochs:03d}] Train') as t:
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = criterion(output, y)

                    # update progress
                    losses.append(loss.item())
                    mean_loss = sum(losses) / len(losses)
                    t.set_postfix_str(f'loss: {mean_loss:.4f}', refresh=False)
                    t.update(n=args.batch_size)


        # save checkpoint only when val_loss decreased
        # early stopping
        # lr decay

    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='head-project')
    p.add_argument('--checkpoint-dir', type=str, default='checkpoint')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-epochs', type=int, default=200)
    p.add_argument('--early-stopping-patience', type=int, default=15)
    p.add_argument('--lr', type=int, default=1e-3)
    p.add_argument('--lr-decay-patience', type=int, default=5)
    p.add_argument('experiment_name', type=str)
    p.add_argument('model_name', type=str)
    p.add_argument('dataset_name', type=str)
    p.add_argument('dataset_path', type=str)
    args = p.parse_args(sys.argv[1:])
    main(args)
