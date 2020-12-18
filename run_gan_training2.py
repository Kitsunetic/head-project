import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_burn as tb
import torch_optimizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from training_loop.data import SingleFileDataset
from training_loop.networks.crnnc import CRNNC_Hardswish

MEANS = torch.tensor([-2.5188, 7.4404, 0.0633, 0.2250, 9.5808, -1.0252], dtype=torch.float32)
STDS = torch.tensor([644.7101, 80.9247, 11.4308, 0.4956, 0.0784, 2.3869], dtype=torch.float32)

plot_idx = -1


class GANDataset(Dataset):
    def __init__(self, X_input: Tensor, X_real: Tensor, window_size: int, means, stds):
        super(GANDataset, self).__init__()

        self.X_input = X_input
        self.X_real = X_real
        assert self.X_input.shape == self.X_real.shape
        self.window_size = window_size
        self.length = self.X_input.shape[0] - self.window_size + 1

        for i in range(3):
            self.X_input[:, i] = (self.X_input[:, i] - means[i]) / stds[i]
            self.X_real[:, i] = (self.X_real[:, i] - means[i]) / stds[i]

        self.X_input = self.X_input.transpose(0, 1)
        self.X_real = self.X_real.transpose(0, 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_input = self.X_input[:, idx:idx + self.window_size]
        x_real = self.X_real[:, idx:idx + self.window_size]
        return x_input, x_real


class ResBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inchannels, channels, kernel_size, stride=1, groups=1):
        super(ResBlock1d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannels, channels, kernel_size, padding=kernel_size // 2, stride=stride, groups=groups),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm1d(channels)
        )
        self.act = nn.LeakyReLU()

        self.conv2 = None
        if inchannels != channels:
            self.conv2 = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride, groups=groups),
                nn.BatchNorm1d(channels)
            )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        if self.conv2 is not None:
            identity = self.conv2(identity)
        x += identity
        x = self.act(x)

        return x


class ConvDetector(nn.Module):
    def __init__(self, block, layers):
        super(ConvDetector, self).__init__()

        self.inchannels = 64

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, self.inchannels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(self.inchannels, channels, 3))
        self.inchannels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inchannels, channels, 3))

        return nn.Sequential(*layers)


class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.Hardswish(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.Hardswish(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x):
        x = self.conv(x)  # B, 3, W
        return x


def plot_results(experiment_name: str, epoch: int, result_dir: Path, history):
    for user in range(7):
        for name in ['Q', 'GAN']:
            x_ = torch.cat(history[f'{name}:{user}:x'])
            y_ = torch.cat(history[f'{name}:{user}:y'])
            p_ = torch.cat(history[f'{name}:{user}:p'])
            x = x_[200:800]
            y = y_[200:800]
            p = p_[200:800]
            X = np.linspace(0, 10, 600)
            plt.figure(figsize=(16, 4))
            plt.title(f'{name}-{experiment_name}-User{user}-Epoch{epoch:03d}')
            for i, title in enumerate(['Yaw', 'Pitch', 'Roll']):
                plt.subplot(1, 3, i + 1)
                plt.plot(X, x[:, i])
                plt.plot(X, p[:, i])
                plt.plot(X, y[:, i])
                plt.ylabel(title + ' (degree)')
                plt.xlabel('Time (sec)')
                plt.legend(['Input', 'Prediction', 'Real'])
            plt.tight_layout()
            plt.savefig(result_dir / f'{name}-{experiment_name}-User{user}-Epoch{epoch:03d}.png')
            plt.close()


def main(args):
    tb.seed_everything(args.seed)

    # Create dataset
    # 각각의 파일별로 열도록 바꿀 것
    ds_trains = [SingleFileDataset(f) for f in sorted(list(args.dataset.glob('*_scene3_0.csv.npz')))]
    ds_tests = [SingleFileDataset(f) for f in sorted(list(args.dataset.glob('*_scene3_1.csv.npz')))]
    dl_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dl_trains = [DataLoader(ds, **dl_kwargs) for ds in ds_trains]
    dl_tests = [DataLoader(ds, **dl_kwargs) for ds in ds_tests]

    # Create model
    Q = CRNNC_Hardswish().cuda()
    G = GModel().cuda()
    D = ConvDetector(ResBlock1d, [2, 2, 2, 2]).cuda()
    q_criterion = nn.MSELoss().cuda()
    g_criterion = nn.MSELoss().cuda()
    d_criterion = nn.BCELoss().cuda()
    q_optimizer = torch_optimizer.RAdam(Q.parameters())
    g_optimizer = torch_optimizer.RAdam(G.parameters())
    d_optimizer = torch_optimizer.RAdam(D.parameters())

    for epoch in range(1, args.epochs + 1):
        # ===============================================================
        #                      Train Loop
        # ===============================================================
        for dsidx, dl in enumerate(dl_trains):
            Q.train()
            G.train()
            D.train()

            # Train Q
            losses = []
            for step, (x, y) in enumerate(dl):
                x = x.cuda()
                y = y.cuda()
                p = Q(x)

                loss = q_criterion(p, y)
                q_optimizer.zero_grad()
                loss.backward()
                q_optimizer.step()

                losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            print(f'[{epoch:03d}/{args.epochs:03d}:{dsidx:d}] Training Q: loss {mean_loss:.4f}')

            # Generate Q's outputs for G's input
            X_input = []
            X_real = []
            with torch.no_grad():
                Q.eval()
                for step, (x, y) in enumerate(dl):
                    p = Q(x.cuda()).cpu()
                    X_input.append(p)
                    X_real.append(y)
            X_input = torch.cat(X_input)  # B, 3
            X_real = torch.cat(X_real)  # B, 3
            ds_gan = GANDataset(X_input, X_real, args.window_size, MEANS, STDS)
            dl_gan = DataLoader(ds_gan, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

            # Train G, D
            losses = [[], [], [], [], [], []]
            for step, (x_input, x_real) in enumerate(dl_gan):
                # Train D
                length = x_input.shape[0]
                x_input = x_input.cuda()
                x_real = x_real.cuda()
                x_fake = G(x_input)
                p_real = D(x_real)
                p_fake = D(x_fake)
                y_real = torch.ones(length, 1, dtype=torch.float32).cuda()
                y_fake = torch.zeros(length, 1, dtype=torch.float32).cuda()
                d_loss_real = d_criterion(p_real, y_real)
                d_loss_fake = d_criterion(p_fake, y_fake)
                d_loss = d_loss_real + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                losses[0].append(d_loss_real.item())
                losses[1].append(d_loss_fake.item())
                losses[2].append(d_loss.item())

                # Train G
                x_fake = G(x_input)
                p_fake = D(x_fake)
                g_loss_real = g_criterion(x_fake, x_real)
                g_loss_fake = d_criterion(p_fake, y_fake)
                g_loss = g_loss_real * 0.2 + g_loss_fake * 0.8
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                losses[3].append(g_loss_real.item())
                losses[4].append(g_loss_fake.item())
                losses[5].append(g_loss.item())

            losses = [sum(l) / len(l) for l in losses]
            print(f'[{epoch:03d}/{args.epochs:03d}:{dsidx:d}] Training D: '
                  f'd_loss: {losses[2]:.4f}, '
                  f'd_loss_real: {losses[0]:.4f}, '
                  f'd_loss_fake: {losses[1]:.4f}')
            print(f'[{epoch:03d}/{args.epochs:03d}:{dsidx:d}] Training G: '
                  f'g_loss: {losses[5]:.4f}, '
                  f'g_loss_real: {losses[3]:.4f}, '
                  f'g_loss_fake: {losses[4]:.4f}')

        # ===============================================================
        #                      Validation Loop
        # ===============================================================
        with torch.no_grad():
            Q.eval()
            G.eval()
            D.eval()
            history = defaultdict(list)
            for dsidx, dl in enumerate(dl_tests):
                # Train Q
                losses = []
                diffs = []
                X_input = []
                X_real = []
                for step, (x_, y_) in enumerate(dl):
                    x = x_.cuda()
                    y = y_.cuda()
                    p = Q(x)
                    p_ = p.cpu()
                    loss = q_criterion(p, y)
                    losses.append(loss.item())
                    diffs.append((y - p).cpu())
                    X_input.append(p_)
                    X_real.append(y_)
                    history[f'Q:{dsidx}:x'].append(x_[:, -1, :])
                    history[f'Q:{dsidx}:y'].append(y_)
                    history[f'Q:{dsidx}:p'].append(p_)

                mean_loss = sum(losses) / len(losses)
                diffs = torch.cat(diffs)  # (B, 3)
                mae = diffs.abs().mean(dim=0)  # (3, ) --> yaw, pitch, roll
                rms = mae.square().sum().div(3).sqrt()  # (1, )
                tile = diffs.abs().mean(dim=1).numpy()  # (B, )
                tile99 = np.percentile(tile, 99)
                print(f'[{epoch:03d}/{args.epochs:03d}:{dsidx:02d}] Validation Q: '
                      f'loss {mean_loss:.4f}, '
                      f'yaw {mae[0].item():.4f}, '
                      f'pitch {mae[1].item():.4f}, '
                      f'roll {mae[2].item():.4f}, '
                      f'rms {rms.item():.4f}, '
                      f'tile99 {tile99:.4f}')

                X_input = torch.cat(X_input)
                X_real = torch.cat(X_real)
                ds_gan = GANDataset(X_input, X_real, args.window_size, MEANS, STDS)
                dl_gan = DataLoader(ds_gan, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

                # Train G, D
                losses = [[], [], [], [], [], []]
                diffs = []
                for step, (x_input, x_real) in enumerate(dl_gan):
                    # Train D
                    length = x_input.shape[0]
                    x_input = x_input.cuda()
                    x_real = x_real.cuda()
                    x_fake = G(x_input)
                    p_real = D(x_real)
                    p_fake = D(x_fake)
                    y_real = torch.ones(length, 1, dtype=torch.float32).cuda()
                    y_fake = torch.zeros(length, 1, dtype=torch.float32).cuda()
                    d_loss_real = d_criterion(p_real, y_real)
                    d_loss_fake = d_criterion(p_fake, y_fake)
                    d_loss = d_loss_real + d_loss_fake
                    losses[0].append(d_loss_real.item())
                    losses[1].append(d_loss_fake.item())
                    losses[2].append(d_loss.item())

                    # Train G
                    x_fake = G(x_input)
                    p_fake = D(x_fake)
                    g_loss_real = g_criterion(x_fake, x_real)
                    g_loss_fake = d_criterion(p_fake, y_fake)
                    g_loss = g_loss_real * 0.1 + g_loss_fake * 0.9
                    losses[3].append(g_loss_real.item())
                    losses[4].append(g_loss_fake.item())
                    losses[5].append(g_loss.item())

                    # Store only the last point
                    x = x_input[:, :, -1].cpu()
                    y = x_real[:, :, -1].cpu()
                    p = x_fake[:, :, -1].cpu()
                    diff = y - p
                    for i in range(3):
                        diff[:, i] *= STDS[i]
                    diffs.append(diff)
                    history[f'GAN:{dsidx}:x'].append(x)
                    history[f'GAN:{dsidx}:y'].append(y)
                    history[f'GAN:{dsidx}:p'].append(p)

                losses = [sum(l) / len(l) for l in losses]
                print(f'[{epoch:03d}/{args.epochs:03d}:{dsidx:02d}] Validation GAN: '
                      f'd_loss: {losses[2]:.4f}, '
                      f'd_loss_real: {losses[0]:.4f}, '
                      f'd_loss_fake: {losses[1]:.4f}, ',
                      f'g_loss: {losses[5]:.4f}, '
                      f'g_loss_real: {losses[3]:.4f}')

                diffs = torch.cat(diffs)  # (B, 3)
                mae = diffs.abs().mean(dim=0)  # (3, ) --> yaw, pitch, roll
                rms = mae.square().sum().div(3).sqrt()  # (1, )
                tile = diffs.abs().mean(dim=1).numpy()  # (B, )
                tile99 = np.percentile(tile, 99)
                print(f'[{epoch:03d}/{args.epochs:03d}:{dsidx:02d}] Validation GAN: '
                      f'yaw {mae[0].item():.4f}, '
                      f'pitch {mae[1].item():.4f}, '
                      f'roll {mae[2].item():.4f}, '
                      f'rms {rms.item():.4f}, '
                      f'tile99 {tile99:.4f}')

        plot_results(args.experiment_name, epoch, args.experiment_path, history)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--window-size', type=int, default=120)
    argparser.add_argument('--epochs', type=int, default=200)
    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--dataset', type=str, required=True)
    argparser.add_argument('--result', type=str, default='results')
    argparser.add_argument('experiment_name', type=str)

    args = argparser.parse_args(sys.argv[1:])
    args.result = Path(args.result)
    dirs = []
    if args.result.is_dir():
        for f in args.result.iterdir():
            if not f.is_dir() or len(f.name) < 6 or not str.isdigit(f.name[:4]):
                continue
            dirs.append(f)
    dirs = sorted(dirs)
    experiment_number = int(dirs[-1].name[:4]) + 1 if dirs else 0
    args.experiment_path = args.result / f'{experiment_number:04d}-{args.experiment_name}'
    args.experiment_path.mkdir(parents=True, exist_ok=True)
    print('Experiment Path:', args.experiment_path)

    args.dataset = Path(args.dataset)

    args.seed = 0
    main(args)
