from dataclasses import dataclass
from torch.nn import functional as F
from model import Classifier

_LABEL2TEXT = {
    0: 'negative',
    1: 'positive'
}

@dataclass
class ClassifyResult:
    """Classification result with label and confidence"""
    label: str
    confidence: float

def predict(text: str, model: Classifier, device: str) -> ClassifyResult:
    tokens = model.tokenizer(
        [text],
        return_tensors='pt',
        padding='max_length', max_length=model.max_length)
    
    pred = model(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))
    pred = F.softmax(pred)[0]
    idx = pred.argmax().item()
    confidence = pred[idx].item()
    
    return ClassifyResult(_LABEL2TEXT[idx], confidence)