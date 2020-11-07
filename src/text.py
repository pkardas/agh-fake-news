from typing import List

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


def get_sentences(text: str) -> List[str]:
    return sent_tokenize(text)


def get_tokens(text: str) -> List[str]:
    return word_tokenize(text)


def get_tokens_without_punctuation(tokens: List[str]) -> List[str]:
    return [
        token.lower()
        for token in tokens
        if token.lower().isalpha()
    ]


def get_tokens_without_stop_words(tokens: List[str]) -> List[str]:
    return [
        token.lower()
        for token in tokens
        if token.lower() not in stopwords.words("english")
    ]
