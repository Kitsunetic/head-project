import os
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pprint_args(args):
    d = args.__dict__
    print('Arguments: ')
    for k, v in d.items():
        print(f' - {k:30}: {v}')
