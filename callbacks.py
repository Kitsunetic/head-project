import math
from pathlib import Path
from typing import AnyStr
from typing import Dict

import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

import utils


class BaseCallback:
    def on_epoch_begin(self, dl: DataLoader, epoch: int, metrics: Dict[torch.tensor]):
        pass

    def on_epoch_end(self, dl: DataLoader, epoch: int, metrics: Dict[torch.tensor]):
        pass

    def on_batch_begin(self, metrics: Dict[torch.tensor]):
        pass

    def on_batch_end(self, metrics: Dict[torch.tensor]):
        pass


class TQDMCallback(BaseCallback):
    def __init__(self, num_epochs: int, desc='[{epoch:03d}/{num_epochs:03d}]', ncols=100):
        super(TQDMCallback, self).__init__()

        self.num_epochs = num_epochs
        self.desc = desc
        self.ncols = ncols

        self.t: tqdm = None

    def on_epoch_begin(self, dl: DataLoader, epoch: int, metrics: Dict[torch.tensor]):
        desc = self.desc.format({'epoch': epoch, 'num_epochs': self.num_epochs})
        self.t = tqdm(total=len(dl), ncols=self.ncols, desc=desc)

    def on_epoch_end(self, dl: DataLoader, epoch: int, metrics: Dict[torch.tensor]):
        if self.t is not None:
            self.t.close()

    def on_batch_end(self, metrics: Dict[torch.tensor]):
        msg = ''
        for k in metrics.keys():
            msg += ' '.join(f'{k} {metrics[k].item()}')
        self.t.set_postfix_str(msg, refresh=False)
        self.t.update()


class SaveCheckpoint(BaseCallback):
    def __init__(self):
        super(SaveCheckpoint, self).__init__()

    def on_epoch_end(self, dl: DataLoader, epoch: int, metrics: Dict[torch.tensor]):
        pass



class SaveCheckpoint:
    def __init__(self, model, optim_encoder, optim_decoder, checkpoint_dir, verbose=False):
        self.model = model.module if type(model) is nn.DataParallel else model
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = math.inf
        self.verbose = verbose

    def __call__(self, epoch, history):
        loss = history['valid']['loss'][-1]
        acc = history['valid']['acc'][-1]
        if loss < self.best_val_loss:
            if self.verbose:
                print('val_loss decreased from', self.best_val_loss, 'to', loss, ': save checkpoint.')
            self.best_val_loss = loss
            checkpoint_path = self.checkpoint_dir / f'ckpt-epoch{epoch:03d}-loss{loss:.4f}-acc{acc:.4f}.pth'
            torch.save({
                'model': self.model.state_dict(),
                'optim_encoder': self.optim_encoder.state_dict(),
                'optim_decoder': self.optim_decoder.state_dict()
            }, checkpoint_path)


class SaveSample:
    def __init__(self, model: nn.Module, ds: Dataset, checkpoint_dir: AnyStr, verbose=False):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose

        # get sample
        x, y, _ = ds[0]
        dx = (x.permute([1, 2, 0]).numpy() * 255.).astype(np.uint8)
        dy = utils.color_label_map(y).numpy()
        imageio.imwrite(self.checkpoint_dir / f'sample_x.png', dx)
        imageio.imwrite(self.checkpoint_dir / f'sample_y.png', dy)

        self.x = x.unsqueeze(0)

    def __call__(self, epoch: int):
        fname = self.checkpoint_dir / f'sample_out-epoch{epoch:03d}.png'
        if self.verbose:
            print('Write sample image to', fname)

        with torch.no_grad():
            self.model.eval()

            x = self.x
            if GPU:
                x = x.cuda()

            out = self.model(x)
            _, out = torch.max(out, dim=1)
            out = utils.color_label_map(out.squeeze()).numpy()
            imageio.imwrite(fname, out)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose

    def __call__(self, history):
        if len(history['valid']['loss']) > self.patience + 1:
            h = history['valid']['loss'][-self.patience:]
            if min(h) == h[0]:
                if self.verbose:
                    print('Stop training because val_loss did not decreased for', self.patience, 'epochs from', h[0])
                return True
        return False


class LRDecaying:
    def __init__(self, optim_encoder, optim_decoder, patience=5, decay_rate=0.5, verbose=False):
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder
        self.patience = patience
        self.decay_rate = decay_rate
        self.verbose = verbose

        self.lr_encoder = next(iter(self.optim_encoder.param_groups.values()))['lr']
        self.lr_decoder = next(iter(self.optim_decoder.param_groups.values()))['lr']

    def __call__(self, history):
        if len(history['valid']['loss']) > self.patience + 1:
            h = history['valid']['loss'][-self.patience:]
            if min(h) == h[0]:
                new_lr_encoder = self.lr_encoder * self.decay_rate
                new_lr_decoder = self.lr_decoder * self.decay_rate
                if self.verbose:
                    print('LR decaying: encoder from', self.lr_encoder, 'to', new_lr_encoder,
                          'decoder from', self.lr_decoder, 'to', new_lr_decoder)

                for p in self.optim_encoder.param_groups:
                    p['lr'] = new_lr_encoder
                for p in self.optim_decoder.param_groups:
                    p['lr'] = new_lr_decoder

                self.lr_encoder = new_lr_encoder
                self.lr_decoder = new_lr_decoder
