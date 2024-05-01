import re


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

