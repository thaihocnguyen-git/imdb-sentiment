"""Training uitility
"""
import time
from statistics import mean
from typing import Dict, Iterable

import torch
from torcheval.metrics import Metric
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot

from metric import get_metrics


def tokenize(
    text,
    label,
    **params):
    """Tokernize and move tensor to device"""
    # turn text to tokens and attention mask
    model = params['model']
    input_tokens = model.tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length' if model.max_length else True,
        max_length=model.max_length)

    # move tensor to target device cpu | cuda
    input_ids = input_tokens["input_ids"].to(model.device)
    attention_mask = input_tokens["attention_mask"].to(model.device)
    label = label.to(model.device)
    return input_ids, attention_mask, label

def train_batch(
    text,
    label,
    metrics: Dict[str, Metric],
    **training_params
    ):
    """Train single batch

    Args:
        text (Union[str, List[str]]): _description_
        label (Union[int, List[int]]): _description_
        tokernizer (PreTrainedTokenizer): _description_
        model (nn.Module): _description_
        criteria (nn.Module): _description_
        optimizer (Optimizer): _description_
        device (str): _description_

    Returns:
        _type_: _description_
    """
    # turn text to tokens and attention mask
    input_ids, attention_mask, label = tokenize(text, label, **training_params)

    # forward and calculate loss
    pred = training_params['model'](input_ids, attention_mask)
    for metric in metrics.values():
        metric.reset()
        metric.update(pred, label)

    loss = training_params['criteria'](pred, label)

    # back probagation
    training_params['optimizer'].zero_grad()
    loss.backward()
    training_params['optimizer'].step()

    return loss.item(), {name: metric.compute().item() for name, metric in metrics.items()}


def train_epoch(
    epoch: int,
    metrics: Dict[str, Metric],
    test=False,
    **training_params
):
    """Training a single epoch.
    """
    print(f"Start training epoch {epoch}")
    losses = []
    execute_time = []
    training_params['model'].train()
    for batch, (text, label) in enumerate(training_params['train_loader']):
        start = time.time()
        loss, metric_values = train_batch(text, label, metrics=metrics, test=test,  **training_params)
        end = time.time()
        
        duration = round(end - start) / 60
        losses.append(loss)
        execute_time.append(duration)
        if batch % training_params.get('print_every', 100) == 1:
            print(f"Batch {batch} of {len(training_params['train_loader'])} , training loss: {loss} in {duration} minutes")
            print(metric_values)
            
        if test:
            break

    print(f"End training epoch {epoch}, loss: {mean(losses)}")
    return mean(losses), metric_values


def validate_epoch(
    epoch: int,
    metrics: Dict[str, Metric],
    **validation_params
):
    """Validation step.
    """
    losses = []
    validation_params['model'].eval()
    for metric in metrics.values():
        metric.reset()

    for text, label in validation_params['validation_loader']:
        input_ids, attention_mask, label = tokenize(text, label, **validation_params)
        pred = validation_params['model'](input_ids, attention_mask)
        loss = validation_params['criteria'](pred, label)
        for metric in metrics.values():
            metric.update(pred, label)
            
        losses.append(loss.item())
    print(f"Validation loss epoch {epoch}: {mean(losses)}")
    result = {f'val_{name}': metric.compute() for name, metric in metrics.items()}
    print(result)
    return mean(losses), result

def optimize(
    metrics: Iterable[str],
    test=False,
    **training_params
):
    """Perform training.
    """
    live_plot = PlotLosses(outputs=[MatplotlibPlot()]) if training_params['live_plot'] else None
    metrics = get_metrics(metrics)
    max_acc = 0
    for epoch in range(training_params['num_epoch']):
        train_loss, train_metrics = train_epoch(epoch, metrics=metrics, test=test, **training_params)
        validation_loss, val_metrics = validate_epoch(epoch, metrics=metrics, **training_params)

        if val_metrics['val_Accuracy'] > max_acc:
            max_acc = val_metrics['val_Accuracy']
            torch.save(training_params['model'].state_dict(), training_params['save_path'])

        if training_params['live_plot']:
            logs = {
                'loss': train_loss,
                'val_loss': validation_loss
            }
            logs.update(train_metrics)
            logs.update(val_metrics)
            live_plot.update(logs)
            live_plot.send()
            
        if test:
            break
