import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_burn as tb
import torch_optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from training_loop.data import CSVSequentialDataset
from training_loop.metrics import HPMetric
from training_loop.networks import get_model_by_name


def main(args):
    tb.seed_everything(args.seed)

    # Create dataset
    csv_files_train = sorted(list(Path(args.dataset).glob('*scene3_0.csv')))
    csv_files_test = sorted(list(Path(args.dataset).glob('*scene3_1.csv')))
    ds_train = [CSVSequentialDataset(f, args.window_size, args.stride) for f in csv_files_train]
    ds_train = tb.data.ChainDataset(*ds_train)
    ds_test = [CSVSequentialDataset(f, args.window_size, args.stride) for f in csv_files_test]
    ds_test = tb.data.ChainDataset(*ds_test)

    # Create model
    model = get_model_by_name(args.network)
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = torch_optimizer.RAdam(model.parameters())

    hp_metric = HPMetric('hp_metric', args.experiment_path)
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
    trainer = tb.Trainer(model, optimizer, metrics, callbacks, ncols=100)
    trainer.fit(ds_train, ds_test, num_epochs=200, batch_size=256, shuffle=True, pin_memory=True)

    # Test
    model.eval()
    torch.set_grad_enabled(False)

    # Load best checkpoint
    ckpt = torch.load(args.experiment_path / 'best-ckpt.pth')
    model.load_state_dict(ckpt['model'])

    # Prediction
    dl = DataLoader(ds_test, batch_size=256, num_workers=16)
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
    plt.plot(T, Y[S:S + L, 0])
    plt.plot(T, P[S:S + L, 0])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Pitch')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'TrainingResult_300-Pitch.png')

    plt.figure(figsize=(height, width))
    plt.plot(T, Y[S:S + L, 0])
    plt.plot(T, P[S:S + L, 0])
    plt.legend(['Real', model.__class__.__name__])
    plt.xlabel('Time (s)')
    plt.ylabel('Degree')
    plt.title('Roll')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'TrainingResult_300-Roll.png')

    # Save loss history
    plt.figure(figsize=(height, width))
    plt.plot(hp_metric.train_history['yaw'])
    plt.plot(hp_metric.train_history['pitch'])
    plt.plot(hp_metric.train_history['roll'])
    plt.plot(hp_metric.train_history['rms'])
    plt.plot(hp_metric.train_history['tile99'])
    plt.legend(['Yaw, Pitch, Roll, RMS, 99Percentile'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Error')
    plt.title('Training Error')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'ErrorPlot-Training.png')

    plt.figure(figsize=(height, width))
    plt.plot(hp_metric.valid_history['yaw'])
    plt.plot(hp_metric.valid_history['pitch'])
    plt.plot(hp_metric.valid_history['roll'])
    plt.plot(hp_metric.valid_history['rms'])
    plt.plot(hp_metric.valid_history['tile99'])
    plt.legend(['Yaw, Pitch, Roll, RMS, 99Percentile'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Error')
    plt.title('Validation Error')
    plt.tight_layout()
    plt.savefig(args.experiment_path / 'ErrorPlot-Validation.png')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--window-size', type=int, default=60)
    argparser.add_argument('--stride', type=int, default=1)
    argparser.add_argument('--network', type=str, required=True)
    argparser.add_argument('--dataset', type=str, required=True)
    argparser.add_argument('--result', type=str, default='results')

    args = argparser.parse_args(sys.argv[1:])
    args.result = Path(args.result)
    dirs = []
    if args.result.is_dir():
        for f in args.result.iterdir():
            if not f.is_dir() or len(f.name) < 6 or not str.isdigit(f.name[:4]):
                continue
            dirs.append(f)
    dirs = sorted(dirs)
    experiment_number = int(dirs[-1][:4]) + 1 if dirs else 0
    experiment_name = f'{experiment_number:04d}-{args.network}-win_{args.window_size}-stride_{args.stride}'
    args.experiment_path = args.result / experiment_name

    args.seed = 0
    main(args)
