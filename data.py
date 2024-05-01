"""Process for IMDB dataset
"""
import re
from typing import Callable, List, Tuple
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

_DATA = None
_DATA_PATH = "stanfordnlp/imdb"

def get_prerpocessor(names) -> List[Callable]:
    """Get preprocess by string"""
    valid_preprocesses = ('remove_stopwords', 'lemma', 'simple_preprocess')
    if any([(name not in valid_preprocesses) for name in names]):
        raise ValueError(f"Invalid text preprocess config. Valid value in {valid_preprocesses}")
    return [globals()[name] for name in names]


def imdb() -> DatasetDict:
    """Get IMDB dataset singletonly."""
    global _DATA
    if not _DATA:
        _DATA = load_dataset(_DATA_PATH)
    return _DATA

def remove_stopwords(text: str) -> str:
    """Remove stopwords in sentence"""
    def is_not_stopword(word: str) -> str:
        return word in stopwords.words('english')
    return ' '.join([word for word in text.split() if is_not_stopword(word)])

def lemma(text: str) -> str:
    """Do the lemmarization for words of the sentences.

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    wnl = WordNetLemmatizer()
    return ' '.join([wnl.lemmatize(word) for word in text.split()])

def simple_preprocess(text: str) -> str:
    """
    Do preprocess steps:
        1. Remove punctuation marks \n
        2. Remove HTML tags \n
        3. Remove URL's \n
        4. Remove characters which are not letters or digits \n
        5. Remove successive whitespaces \n
        6. Convert the text to lower case \n
        7. strip whitespaces from the beginning and the end of the reviews \n

    Args:
        text (str): Original text.

    Returns:
        str: _description_
    """
    text = re.sub(r'[,\.!?:()"]', '', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"'ll", ' will', text)
    text = re.sub(r'https?\S+\s', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

class IMBDWrapped(Dataset):
    """The IMDB dataset wit preprocessing."""
    def __init__(self,
                 dataset: Dataset,
                 text_processors: List[Callable]) -> None:
        super().__init__()
        self._data = dataset
        self._preprocessors = text_processors


    def __len__(self):
        return len(self._data)

    def _preprocess(self, data, processor):
        if isinstance(data, str):
            return processor(data)

        if isinstance(data, List):
            return [processor(text) for text in data]

        return data

    def __getitem__(self, index) -> Tuple[str, int]:
        item = self._data[index]
        text, label = item['text'], item['label']
        for processor in self._preprocessors:
            text = self._preprocess(text, processor)
        return text, label
