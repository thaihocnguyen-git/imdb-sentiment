from typing import Callable, Dict, Iterable
from torch.utils.data import DataLoader

from metric import get_metrics
from model import Classifier
from train import tokenize

def test(
    model: Classifier,
    ds_loader: DataLoader,
    metrics: Iterable[str],
    metric_callbacks: Iterable[Callable]=(),
    verbose=False) -> Dict[str,float]:
    """Test the model.

    Args:
        model (Classifier): The model to test
        ds_loader (DataLoader): 
        metrics (Iterable[Metric]): _description_
        metric_callbacks (Callable], optional): _description_. Defaults to None.

    Returns:
        Iterable[float]: _description_
    """
    if not metrics:
        raise ValueError('Please choose a metric to test')

    metrics = get_metrics(metrics)

    print(f"Start testing model, batch_size={ds_loader.batch_size}, number of batch {len(ds_loader)}")
    model.eval()
    for batch, (text, label) in enumerate(ds_loader):
        input_ids, mask, label = tokenize(text=text, label=label, tokenize=model.tokenizer, model=model)
        pred = model(input_ids, mask)
        if verbose:
            print(f'Evaluted batch {batch}')

        for name, metric in metrics.items():
            metric.update(pred, label)
            if verbose:
                print_metric(name, metric.compute())

    result = {name: metric.compute().item() for name, metric in metrics.items()}

    if metric_callbacks:
        for func in metric_callbacks:
            for metric, value in result.items():
                func(metric, value)

    return result

def print_metric(name, value):
    """Print the metrics detail to console."""
    print(f"{name}: {value}")