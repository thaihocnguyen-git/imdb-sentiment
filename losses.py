"""Loss functions"""
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
def get_loss(name: str='CrossEntropyLoss', **kwargs) -> nn.Module:
    """Get loss function"""
    if name == 'CrossEntropyLoss':
        return CrossEntropyLoss(**kwargs)
    if name == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss(**kwargs)

    raise ValueError('Invalid loss name. Expected CrossEntropyLoss | BCEWithLogitsLoss')