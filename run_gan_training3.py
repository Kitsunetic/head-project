"""
Q모델을 학습하면서 GAN의 G, D 모델도 함께 학습시켜보자

Q모델은 shuffle을 안하면 학습이 느려진다는 특징 --> 일반적인 time series 모델들이 공유하는 문제일 듯?
GAN 모델들은 학습이 잘 안됨
"""
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
from torch.utils.data import DataLoader, Dataset

from training_loop.networks.resnet import ResBlock1d

# MEANS = torch.tensor([-2.5188, 7.4404, 0.0633, 0.2250, 9.5808, -1.0252], dtype=torch.float32)
# STDS = torch.tensor([644.7101, 80.9247, 11.4308, 0.4956, 0.0784, 2.3869], dtype=torch.float32)
# MEANS = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)
# STDS = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)
MEANS = torch.tensor([-1.6627, 8.2190, 0.5204, 0.3034, 9.5687, -1.1618], dtype=torch.float32)
STDS = torch.tensor([24.1827, 8.8223, 3.1585, 0.6732, 0.2772, 1.5191], dtype=torch.float32)


class GANDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        super(GANDataset, self).__init__()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.X = (self.X - MEANS.reshape(1, 1, 6)) / STDS.reshape(1, 1, 6)
        self.Y = (self.Y - MEANS[:3].reshape(1, 1, 3)) / STDS[:3].reshape(1, 1, 3)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # B, S, C
        y = self.Y[idx]
        return x, y


class ConvDetector(nn.Module):
    def __init__(self, block, layers):
        super(ConvDetector, self).__init__()

        self.inchannels = 64

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, self.inchannels, 7, stride=2, padding=3, padding_mode='replicate', bias=False),
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
        x = x.transpose(1, 2)
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


class CRNNC_GAN(nn.Module):
    def __init__(self):
        super(CRNNC_GAN, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2, padding_mode='replicate'),
            nn.BatchNorm1d(64),
            nn.Hardswish(),
            ResBlock1d(64, 64, 3, Activation=nn.Hardswish)
        )

        self.rnn = nn.RNN(input_size=64,
                          hidden_size=64,
                          num_layers=4,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 64, 3, Activation=nn.Hardswish),
            ResBlock1d(64, 128, 3, Activation=nn.Hardswish),
            ResBlock1d(128, 128, 3, Activation=nn.Hardswish),
            ResBlock1d(128, 256, 3, Activation=nn.Hardswish),
            nn.Conv1d(256, 3, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_in(x)  # B, 6, S

        x = x.transpose(1, 2)  # B, S, 6
        outs, _ = self.rnn(x)  # B, S, 128
        x = outs.transpose(1, 2)  # B, C, S

        x = self.conv_out(x)  # B, C, S
        x = x.transpose(1, 2)  # B, S, C

        return x


def plot_results(experiment_name: str, epoch: int, name: str, result_dir: Path, history):
    x_ = torch.cat(history[f'{name}:x'])
    y_ = torch.cat(history[f'{name}:y'])
    p_ = torch.cat(history[f'{name}:p'])
    x = x_[200:800]
    y = y_[200:800]
    p = p_[200:800]
    X = np.linspace(0, 10, 600)
    plt.figure(figsize=(16, 4))
    plt.title(f'{name}-{experiment_name}-Epoch{epoch:03d}')
    for i, title in enumerate(['Yaw', 'Pitch', 'Roll']):
        plt.subplot(1, 3, i + 1)
        plt.plot(X, x[:, i])
        plt.plot(X, p[:, i])
        plt.plot(X, y[:, i])
        plt.ylabel(title + ' (degree)')
        plt.xlabel('Time (sec)')
        plt.legend(['Input', 'Prediction', 'Real'])
    plt.tight_layout()
    plt.savefig(result_dir / f'{name}-{experiment_name}-Epoch{epoch:03d}.png')
    plt.close()


def annot_min(x, y, name, ax=None, xpos=None):
    if xpos is None:
        xpos = np.argmin(y)
        xmax = x[xpos]
        ymax = min(y)
    else:
        xmax = x[xpos]
        ymax = y[xpos]
    text = f"epoch={xmax:.3f}, {name} {ymax:.3f}"
    if not ax:
        ax = plt.gca()
    # arrowprops = dict(facecolor='black', shrink=0.7)
    arrowprops = dict(arrowstyle='->', connectionstyle="angle,angleA=0,angleB=60")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax - 5, ymax - 2), arrowprops=arrowprops)

    return xpos


def plot_error_history(args, name, history):
    epoch = len(history[f'{name}:loss'])
    X = np.linspace(0, epoch, epoch)

    plt.figure(figsize=(6, 6))
    plt.plot(X, history[f'{name}:yaw'])
    plt.plot(X, history[f'{name}:pitch'])
    plt.plot(X, history[f'{name}:roll'])
    plt.plot(X, history[f'{name}:rms'])
    plt.plot(X, history[f'{name}:tile99'])
    plt.legend(['Yaw', 'Pitch', 'Roll', 'RMS', '99%ile'])
    xpos = annot_min(X, np.array(history[f'{name}:tile99']), '99tile')
    annot_min(X, np.array(history[f'{name}:yaw']), 'yaw', xpos=xpos)
    annot_min(X, np.array(history[f'{name}:pitch']), 'pitch', xpos=xpos)
    annot_min(X, np.array(history[f'{name}:roll']), 'roll', xpos=xpos)
    annot_min(X, np.array(history[f'{name}:rms']), 'rms', xpos=xpos)
    plt.xlabel('Epochs')
    plt.ylabel('Error (Degree)')
    plt.ylim(0, 25)
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'error_plot-Q.png')


def main(args):
    tb.seed_everything(args.seed)
    plt.switch_backend('agg')  # matplotlib을 cli에서 사용

    # 데이터셋 불러오기
    data_train = np.load('data/1116/train-win_120-GAN.npz')
    data_test = np.load('data/1116/test-win_120-GAN.npz')
    ds_train = GANDataset(data_train['X'], data_train['Y'])
    ds_test = GANDataset(data_test['X'], data_test['Y'])
    dl_kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
    dl_test = DataLoader(ds_test, **dl_kwargs, shuffle=False)

    # Create model
    G = CRNNC_GAN().cuda()
    D = ConvDetector(ResBlock1d, [2, 2, 2, 2]).cuda()
    g_criterion = nn.MSELoss().cuda()
    d_criterion = nn.BCELoss().cuda()
    g_optimizer = torch_optimizer.RAdam(G.parameters())
    d_optimizer = torch_optimizer.RAdam(D.parameters())

    div_means = MEANS[:3].reshape(1, 3, 1)
    div_stds = STDS[:3].reshape(1, 3, 1)
    for epoch in range(1, args.epochs + 1):
        losses = [[], [], [], [], [], []]
        G.train()
        D.train()
        for x_input_, x_real_ in dl_train:
            # D
            x_input = x_input_.cuda()
            x_real = x_real_.cuda()
            x_fake = G(x_input)
            p_real = D(x_real)
            p_fake = D(x_fake)
            y_real = torch.ones(x_real.shape[0], 1, dtype=torch.float32).cuda()
            y_fake = torch.zeros(x_fake.shape[0], 1, dtype=torch.float32).cuda()
            d_loss_real = d_criterion(p_real, y_real)
            d_loss_fake = d_criterion(p_fake, y_fake)
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            losses[0].append(d_loss.item())
            losses[1].append(d_loss_real.item())
            losses[2].append(d_loss_fake.item())

            # G
            x_fake = G(x_input) # B, S, C
            p_fake = D(x_fake)
            g_loss_real = g_criterion(x_fake[:, -1:, :], x_real[:, -1:, :])
            g_loss_fake = d_criterion(p_fake, y_fake)
            g_loss = g_loss_real * 0.05 + g_loss_fake * 0.95
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            losses[3].append(g_loss.item())
            losses[4].append(g_loss_real.item())
            losses[5].append(g_loss_fake.item())

        losses = [sum(l) / len(l) for l in losses]
        print(f'[{epoch:03d}/{args.epochs:03d}] Train GAN: '
              f'd_loss: {losses[0]:.4f}, '
              f'd_loss_real: {losses[1]:.4f}, '
              f'd_loss_fake: {losses[2]:.4f}, '
              f'g_loss: {losses[3]:.4f}, '
              f'g_loss_fake: {losses[5]:.4f}')

        G.eval()
        D.eval()
        with torch.no_grad():
            losses = [[], [], [], [], [], []]
            X, Y, P = [], [], []
            diffs = []
            for x_input_, x_real_ in dl_test:
                # D
                x_input = x_input_.cuda()
                x_real = x_real_.cuda()
                x_fake = G(x_input)
                x_fake_ = x_fake.cpu()
                p_real = D(x_real)
                p_fake = D(x_fake)
                y_real = torch.ones(x_real.shape[0], 1, dtype=torch.float32).cuda()
                y_fake = torch.zeros(x_fake.shape[0], 1, dtype=torch.float32).cuda()
                d_loss_real = d_criterion(p_real, y_real)
                d_loss_fake = d_criterion(p_fake, y_fake)
                d_loss = d_loss_real + d_loss_fake
                losses[0].append(d_loss.item())
                losses[1].append(d_loss_real.item())
                losses[2].append(d_loss_fake.item())

                # G
                x_fake = G(x_input)
                p_fake = D(x_fake)
                g_loss_real = g_criterion(x_fake[:, -1:, :], x_real[:, -1:, :])
                g_loss_fake = d_criterion(p_fake, y_fake)
                g_loss = g_loss_real * 0.05 + g_loss_fake * 0.95
                losses[3].append(g_loss.item())
                losses[4].append(g_loss_real.item())
                losses[5].append(g_loss_fake.item())

                x = x_input_.transpose(1, 2)[:, :3, -18:] * div_stds + div_means
                y = x_real_.transpose(1, 2)[:, :3, -18:] * div_stds + div_means
                p = x_fake_.transpose(1, 2)[:, :3, -18:] * div_stds + div_means
                diffs.append(y - p)
                X.append(torch.flatten(x, 0, 1))
                Y.append(torch.flatten(y, 0, 1))
                P.append(torch.flatten(p, 0, 1))

            losses = [sum(l) / len(l) for l in losses]
            print(f'[{epoch:03d}/{args.epochs:03d}] Validate GAN: '
                  f'd_loss: {losses[0]:.4f}, '
                  f'd_loss_real: {losses[1]:.4f}, '
                  f'd_loss_fake: {losses[2]:.4f}, '
                  f'g_loss: {losses[3]:.4f}, '
                  f'g_loss_fake: {losses[5]:.4f}')

            diffs = torch.cat(diffs)  # (B, 3, 18)
            mae = diffs.abs().mean(dim=[0, 2])  # (3, ) --> yaw, pitch, roll
            rms = mae.square().sum().div(3).sqrt()  # (1, )
            tile = diffs.square().mean(dim=[1, 2]).sqrt().numpy()  # (B, 18)
            tile99 = np.percentile(tile, 99)
            print(f'[{epoch:03d}/{args.epochs:03d}] Validate GAN: '
                  f'yaw {mae[0].item():.4f}, '
                  f'pitch {mae[1].item():.4f}, '
                  f'roll {mae[2].item():.4f}, '
                  f'rms {rms.item():.4f}, '
                  f'tile99 {tile99:.4f}')

            X = torch.cat(X)  # (L, 3)
            Y = torch.cat(Y)
            P = torch.cat(P)
            np.savez_compressed(args.experiment_path / f'data-epoch{epoch}.npz', X=X, Y=Y, P=P)

            # TODO plot


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=200)
    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--result', type=str, default='results')
    argparser.add_argument('--comment', type=str, default='')

    args = argparser.parse_args(sys.argv[1:])
    args.result = Path(args.result)
    dirs = []
    if args.result.is_dir():
        for f in args.result.iterdir():
            if not f.is_dir() or len(list(f.iterdir())) == 0 or not str.isdigit(f.name[:4]):
                continue
            dirs.append(f)
    dirs = sorted(dirs)
    experiment_number = int(dirs[-1].name[:4]) + 1 if dirs else 0
    comment = f'-{args.comment}' if args.comment else ''
    args.experiment_path = args.result / f'{experiment_number:04d}{comment}'
    args.experiment_path.mkdir(parents=True, exist_ok=True)
    print('Experiment Path:', args.experiment_path)

    args.seed = 0
    main(args)
