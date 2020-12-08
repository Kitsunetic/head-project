import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_burn as tb
import torch_optimizer
from torch import Tensor
from torch.utils.data import DataLoader

from training_loop.data import SingleFileDataset

MEANS = torch.tensor([-2.5188, 7.4404, 0.0633, 0.2250, 9.5808, -1.0252], dtype=torch.float32)
STDS = torch.tensor([644.7101, 80.9247, 11.4308, 0.4956, 0.0784, 2.3869], dtype=torch.float32)

plot_idx = -1


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


class CRNNC_M2M(nn.Module):
    def __init__(self, Activation=nn.LeakyReLU):
        super(CRNNC_M2M, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, groups=2),
            nn.BatchNorm1d(64),
            Activation(),
            ResBlock1d(64, 64, 3)
        )

        self.rnn = nn.RNN(input_size=64,
                          hidden_size=64,
                          num_layers=4,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)

        self.conv_out = nn.Sequential(
            ResBlock1d(64, 128, 3),
            ResBlock1d(128, 256, 3),
            nn.Conv1d(256, 3, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # B, S, 6 --> B, 6, S
        x = self.conv_in(x)  # B, 6, S
        x = x.transpose(1, 2)  # B, S, 6

        outs, _ = self.rnn(x)  # B, S, 128
        x = outs.transpose(1, 2)  # B, C, S
        x = self.conv_out(x)  # B, 3, S
        x = x.transpose(1, 2)  # B, S, 3

        return x


def plot_results(epoch: int, result_dir: Path, x_input: Tensor, x_real: Tensor, x_fake: Tensor):
    global plot_idx

    x_input = x_input.detach().cpu()
    x_real = x_real.detach().cpu()
    x_fake = x_fake.detach().cpu()

    plot_idx = random.randint(0, x_input.shape[0])

    X = np.linspace(0, x_input.shape[1] / 60, x_input.shape[1])
    titles = [f'Yaw - {epoch:03d}', f'Pitch - {epoch:03d}', f'Roll - {epoch:03d}']

    plt.figure(figsize=(16, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(X, x_real[plot_idx, :, i] * STDS[i] + MEANS[i])
        plt.plot(X, x_fake[plot_idx, :, i] * STDS[i] + MEANS[i])
        plt.title(titles[i])
        plt.ylabel('Degree')
        plt.xlabel('Time (s)')
        plt.legend(['Real', 'Fake'])
        plt.ylim(-100, 100)
        plt.tight_layout()
        plt.savefig(result_dir / f'sample-{epoch:03d}.png')
    plt.close()


def main(args):
    tb.seed_everything(args.seed)

    # Create dataset
    ds_train = SingleFileDataset(Path(args.dataset) / f'train-win_{args.window_size}-m2m.npz', means=MEANS, stds=STDS)
    ds_test = SingleFileDataset(Path(args.dataset) / f'test-win_{args.window_size}-m2m.npz', means=MEANS, stds=STDS)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, num_workers=6, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, num_workers=6, pin_memory=True)

    # Create model
    G = CRNNC_M2M(Activation=nn.Hardswish).cuda()
    D = ConvDetector(ResBlock1d, [2, 2, 2, 2]).cuda()
    g_criterion = nn.MSELoss().cuda()
    d_criterion = nn.BCELoss().cuda()
    g_optimizer = torch_optimizer.RAdam(G.parameters())
    d_optimizer = torch_optimizer.RAdam(D.parameters())

    for epoch in range(1, args.epochs + 1):
        # ===============================================================
        #                      Train Loop
        # ===============================================================
        G.train()
        D.train()
        total_steps = len(dl_train)
        for step, (x_input, x_real) in enumerate(dl_train, 1):
            x_input = x_input.cuda()
            x_real = x_real.cuda()
            x_fake = G(x_input)

            pred_real = D(x_real)
            pred_fake = D(x_fake)
            y_real = torch.ones(pred_real.shape[0], 1, dtype=torch.float32).cuda()
            y_fake = torch.zeros(pred_real.shape[0], 1, dtype=torch.float32).cuda()

            d_loss_real = d_criterion(pred_real, y_real)
            d_loss_fake = d_criterion(pred_fake, y_fake)
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            x_fake = G(x_input)
            pred_fake = D(x_fake)
            g_loss_real = g_criterion(x_fake, x_real)
            g_loss_fake = d_criterion(pred_fake, y_fake)
            g_loss = g_loss_real * 0.1 + g_loss_fake * 0.9
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if step % 100 == 0 or step == total_steps:
                print(f'Train Epoch[{epoch:03d}/{args.epochs:03d}] Step [{step:03d}/{total_steps:03d}] '
                      f'd_loss: {d_loss.item():.4f}, '
                      f'd_loss_real: {d_loss_real.item():.4f}, '
                      f'd_loss_fake: {d_loss_fake:.4f} '
                      f'g_loss: {g_loss.item():.4f}')

        # ===============================================================
        #                      Validation Loop
        # ===============================================================
        with torch.no_grad():
            G.eval()
            D.eval()
            total_steps = len(dl_test)
            for step, (x_input, x_real) in enumerate(dl_test, 1):
                x_input = x_input.cuda()
                x_real = x_real.cuda()
                x_fake = G(x_input)

                pred_real = D(x_real)
                pred_fake = D(x_fake)
                y_real = torch.ones(pred_real.shape[0], 1, dtype=torch.float32).cuda()
                y_fake = torch.zeros(pred_real.shape[0], 1, dtype=torch.float32).cuda()

                d_loss_real = d_criterion(pred_real, y_real)
                d_loss_fake = d_criterion(pred_fake, y_fake)
                d_loss = d_loss_real + d_loss_fake

                x_fake = G(x_input)
                pred_fake = D(x_fake)
                g_loss_real = g_criterion(x_fake, x_real)
                g_loss_fake = d_criterion(pred_fake, y_fake)
                g_loss = g_loss_real * 0.1 + g_loss_fake * 0.9

                if step % 100 == 0 or step == total_steps:
                    print(f'Valid Epoch[{epoch:03d}/{args.epochs:03d}] Step [{step:02d}/{total_steps:02d}] '
                          f'd_loss: {d_loss.item():.4f}, '
                          f'd_loss_real: {d_loss_real.item():.4f}, '
                          f'd_loss_fake: {d_loss_fake:.4f} '
                          f'g_loss: {g_loss.item():.4f}')

            plot_results(epoch, args.experiment_path, x_input, x_real, x_fake)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--window-size', type=int, default=60)
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

    args.seed = 0
    main(args)
