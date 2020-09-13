import torch.nn as nn

from .SimpleErrorModel import SimpleErrorModel

MODEL_NAMES = {
    'simpleerrormodel': SimpleErrorModel
}


def from_name(model_name, **kwargs) -> nn.Module:
    model_name = model_name.lower()
    if model_name in MODEL_NAMES:
        model_fn = MODEL_NAMES[model_name]
        model = model_fn(**kwargs)
        return model
    else:
        raise NotImplementedError(f'Model not fount: {model_name}')
