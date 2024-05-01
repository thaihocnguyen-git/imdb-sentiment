"""Optimizers"""
from torch.optim import Optimizer, Adam, AdamW, SGD
def get_optimizer(params, **kwargs) -> Optimizer:
    """Get optimizer.
    """
    name = kwargs['name']
    del kwargs['name']
    if name == 'Adam':
        return Adam(params=params, **kwargs)

    if name == 'AdamW':
        return AdamW(params=params, **kwargs)

    if name == 'SGD':
        return SGD(params=params, **kwargs)

    raise ValueError('Invalid optimizer name. Expected "Adam" | "AdamW" | "SDG"')

