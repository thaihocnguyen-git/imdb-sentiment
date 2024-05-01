
from typing import Iterable, Dict
from torcheval.metrics import Metric, MulticlassF1Score, MulticlassAccuracy

_F1 = 'F1'
_ACCURACY = "Accuracy"
def get_metrics(names: Iterable[str]) -> Dict[str, Metric]:
    """Factory for validation metrics.
    """
    def get_metric(name):
        if name == _F1:
            return MulticlassF1Score()

        if name == _ACCURACY:
            return MulticlassAccuracy()

        raise ValueError(f'Invalid metrics name, expect {[_F1, _ACCURACY]}')
    return {name: get_metric(name) for name in names}