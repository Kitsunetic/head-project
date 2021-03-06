import argparse
import sys
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_burn as tb
import torch_optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SingleFileDataset
from metrics import HPMetric
from networks import get_model_by_name


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
    arrowprops = dict(arrowstyle='->', connectionstyle="angle,angleA=0,angleB=60")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax - 5, ymax - 2), arrowprops=arrowprops)

    return xpos


def main(args):
    tb.seed_everything(args.seed)
    plt.switch_backend('agg')

    # Create dataset
    ds_train = SingleFileDataset(Path(args.dataset) / f'train-win_{args.window_size}.npz')
    ds_test = SingleFileDataset(Path(args.dataset) / f'test-win_{args.window_size}.npz')

    # Create model
    model = get_model_by_name(args.network)
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = torch_optimizer.RAdam(model.parameters())

    hp_metric = HPMetric('hp_metric', args.experiment_path, 'history.log')
    metrics = [
        tb.metrics.ModuleMetric(criterion, 'loss'),
        hp_metric
    ]
    callbacks = [
        # tb.callbacks.EarlyStopping(metrics[0]),
        tb.callbacks.LRDecaying(optimizer, metrics[0], patience=3),
        tb.callbacks.SaveCheckpoint({'model': model}, metrics[0], args.experiment_path, 'best-ckpt.pth')
    ]

    # Training
    trainer = tb.Trainer(model, optimizer, metrics, callbacks, ncols=100, cpus=args.cpus)
    trainer.fit(ds_train, ds_test, num_epochs=args.epochs, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Test
    model.eval()
    torch.set_grad_enabled(False)

    # Load best checkpoint
    ckpt = torch.load(args.experiment_path / 'best-ckpt.pth')
    model.load_state_dict(ckpt['model'])

    # Prediction
    dl = DataLoader(ds_test, batch_size=args.batch_size, num_workers=16)
    inputs, targets, preds = [], [], []
    with tqdm(total=len(dl), position=0, ncols=100) as t:
        for x, y in dl:
            pred = model(x.cuda()).cpu()
            preds.append(pred)

            inputs.append(x)
            targets.append(y)

            t.update()
    X = torch.cat(inputs)
    Y = torch.cat(targets)
    P = torch.cat(preds)

    # Save prediction as file
    np.save(args.experiment_path / 'result-X.npy', X.numpy())
    np.save(args.experiment_path / 'result-Y.npy', Y.numpy())
    np.save(args.experiment_path / 'result-P.npy', P.numpy())

    # Graph parameters
    height = 6
    width = 4

    # Plot graph as image file
    S = 600
    L = 300
    T = np.linspace(0, L / 60, L)
    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 0])
    plt.plot(T, P[S:S + L, 0])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Yaw')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'TrainingResult_300-Yaw.png')

    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 1])
    plt.plot(T, P[S:S + L, 1])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Pitch')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'TrainingResult_300-Pitch.png')

    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 2])
    plt.plot(T, P[S:S + L, 2])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Roll')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'TrainingResult_300-Roll.png')

    S = 600
    L = 600
    T = np.linspace(0, L / 60, L)
    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 0])
    plt.plot(T, P[S:S + L, 0])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Yaw')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'TrainingResult_600-Yaw.png')

    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 1])
    plt.plot(T, P[S:S + L, 1])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Pitch')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'TrainingResult_600-Pitch.png')

    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 2])
    plt.plot(T, P[S:S + L, 2])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Roll')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'TrainingResult_600-Roll.png')

    S = 600
    L = 3000
    T = np.linspace(0, L / 60, L)
    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 0])
    plt.plot(T, P[S:S + L, 0])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Yaw')
    plt.tight_layout()
    plt.savefig(args.experiment_path / f'TrainingResult_{L}-Yaw.png')

    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 1])
    plt.plot(T, P[S:S + L, 1])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Pitch')
    plt.tight_layout()
    plt.savefig(args.experiment_path / f'TrainingResult_{L}-Pitch.png')

    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 2])
    plt.plot(T, P[S:S + L, 2])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Roll')
    plt.tight_layout()
    plt.savefig(args.experiment_path / f'TrainingResult_{L}-Roll.png')

    # Save loss history
    epochs_x = np.array(list(range(1, args.epochs + 1)), dtype=np.int)
    plt.figure(figsize=(height, width))
    plt.plot(epochs_x, hp_metric.train_history['yaw'])
    plt.plot(epochs_x, hp_metric.train_history['pitch'])
    plt.plot(epochs_x, hp_metric.train_history['roll'])
    plt.plot(epochs_x, hp_metric.train_history['rms'])
    plt.plot(epochs_x, hp_metric.train_history['tile99'])
    plt.legend(['Yaw', 'Pitch', 'Roll', 'RMS', '99Percentile'])
    xpos = annot_min(epochs_x, np.array(hp_metric.train_history['tile99']), '99tile')
    annot_min(epochs_x, np.array(hp_metric.train_history['yaw']), 'yaw', xpos=xpos)
    annot_min(epochs_x, np.array(hp_metric.train_history['pitch']), 'pitch', xpos=xpos)
    annot_min(epochs_x, np.array(hp_metric.train_history['roll']), 'roll', xpos=xpos)
    annot_min(epochs_x, np.array(hp_metric.train_history['rms']), 'rms', xpos=xpos)
    plt.xlabel('Epoch (Numbers)')
    plt.ylabel('Mean Error (Degree)')
    plt.ylim(0, 25)
    plt.title(f'Training Error ({model.__class__.__name__})')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'ErrorPlot-Training.png')

    plt.figure(figsize=(height, width))
    plt.plot(epochs_x, hp_metric.valid_history['yaw'])
    plt.plot(epochs_x, hp_metric.valid_history['pitch'])
    plt.plot(epochs_x, hp_metric.valid_history['roll'])
    plt.plot(epochs_x, hp_metric.valid_history['rms'])
    plt.plot(epochs_x, hp_metric.valid_history['tile99'])
    plt.legend(['Yaw', 'Pitch', 'Roll', 'RMS', '99Percentile'])
    xpos = annot_min(epochs_x, np.array(hp_metric.valid_history['tile99']), '99tile')
    annot_min(epochs_x, np.array(hp_metric.valid_history['yaw']), 'yaw', xpos=xpos)
    annot_min(epochs_x, np.array(hp_metric.valid_history['pitch']), 'pitch', xpos=xpos)
    annot_min(epochs_x, np.array(hp_metric.valid_history['roll']), 'roll', xpos=xpos)
    annot_min(epochs_x, np.array(hp_metric.valid_history['rms']), 'rms', xpos=xpos)
    plt.xlabel('Epoch (Numbers)')
    plt.ylabel('Mean Error (Degree)')
    plt.ylim(0, 25)
    plt.title(f'Validation Error ({model.__class__.__name__})')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'ErrorPlot-Validation.png')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--window-size', type=int, default=60)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--cpus', type=int, default=cpu_count())
    argparser.add_argument('--network', type=str, required=True)
    argparser.add_argument('--dataset', type=str, required=True)
    argparser.add_argument('--result', type=str, default='results')
    argparser.add_argument('--comment', type=str, default='')

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
    comment = f'-{args.comment}' if args.comment else ''
    experiment_name = f'{experiment_number:04d}-{args.network}-win_{args.window_size}' \
                      f'-epoch_{args.epochs:02d}-batch_size_{args.batch_size}{comment}'
    args.experiment_path = args.result / experiment_name
    print('Experiment Path:', args.experiment_path)

    args.seed = 0
    main(args)
