"""Get model and tokenizer"""

import os
from typing import Tuple, Iterable
from pathlib import Path
import torch
from transformers import (PreTrainedTokenizer,
                          PreTrainedModel,
                          BertModel,
                          BertTokenizer,
                          RobertaTokenizerFast,
                          RobertaModel)
from torch.nn import Module, ReLU, Linear, Dropout, Sequential
BERT_BASE_UNCASED = 'google-bert/bert-base-uncased'
BERT_BASE_CASED = 'google-bert/bert-base-cased'
ROBERTA_BASE = 'FacebookAI/roberta-base'
BASE_HIDDEN_DIM = 768

def _get_classifier(
    input_dim: int,
    num_labels: str,
    hidden_dims: Iterable[int] = (),
    dropout: float=0.0,
    activation: Module=ReLU()
    ):
    """Create MLP networks, which will receive output of the base model to classify input to logits"""
    in_dims = [input_dim] + hidden_dims
    out_dims = hidden_dims + [num_labels]
    layers = []
    for in_dim, out_dim in zip(in_dims, out_dims):
        layers.append(Linear(in_dim, out_dim))
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers.append(activation)

    layers = layers[:-1] #the last layers don't need activation
    return Sequential(*layers)


class Classifier(Module):
    """Classifier model
    Args:
        Module (_type_): _description_
    """
    _classifier_fn = "classifier.pt"
    _bert_base_fn = 'base.pt'
    def __init__(self, base: PreTrainedModel, classifier: Module, tokenizer, max_length, num_classes) -> None:
        super().__init__()
        self._base = base
        self._classifier = classifier
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes

    def load_checkpoint(self, ckpt_path: str) -> None:
        """Load checkpoint from saved path is existed."""
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path)
            self.load_state_dict(state_dict)
            print(f'Loaded checkpoint from {ckpt_path}')


    def forward(self, input_ids, attention_mask):
        """Forward"""
        x = self._base(input_ids, attention_mask).pooler_output
        x = self._classifier(x)
        return x

    @property
    def device(self):
        """The device (cpu | cuda)"""
        return next(self.parameters()).device

def get_model(**kwargs) -> Tuple[Classifier, PreTrainedTokenizer]:
    """Get the model and classifier"""
    if kwargs['name'] in (BERT_BASE_UNCASED, BERT_BASE_CASED):
        base_model, tokenizer = (
            BertModel.from_pretrained(kwargs['name']),
            BertTokenizer.from_pretrained(kwargs['name'])
        )

    elif kwargs['name'] == ROBERTA_BASE:
        base_model, tokenizer = (
            RobertaModel.from_pretrained(ROBERTA_BASE),
            RobertaTokenizerFast.from_pretrained(ROBERTA_BASE)
        )

    else:
        raise ValueError(f'Invalid model name. Expect value in `{BERT_BASE_CASED}` | `{BERT_BASE_UNCASED}` | `{ROBERTA_BASE}`')
    for p in base_model.parameters():
        p.requires_grad = False
    for p in base_model.pooler.parameters():
        p.requires_grad = True

    classifier = _get_classifier(BASE_HIDDEN_DIM, **kwargs['classifier'])
    model = Classifier(base_model
                       , classifier,
                       tokenizer,
                       max_length=kwargs['tokenizer']['max_length'],
                       num_classes=kwargs['classifier']['num_labels'])
    return model
