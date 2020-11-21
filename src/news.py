from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from datetime import date
from os import path
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

from src.features import get_sentiment
from src.models import Sentiment

logger = logging.getLogger()


class Text:
    def __init__(self, text: str):
        self.text = clear_text(text)

    def __str__(self) -> str:
        return self.text

    def __add__(self, other) -> Text:
        if not isinstance(other, Text):
            return self
        return Text(self.text + ' ' + other.text)

    @property
    def sentences(self) -> List[str]:
        return sent_tokenize(self.text)

    @property
    def tokens(self) -> List[str]:
        """
        Returns list of lowercase tokens without stop words.
        """
        tokens = word_tokenize(self.text)
        return [
            token.lower()
            for token in tokens
            if token.isalpha() and token.lower() not in stopwords.words("english")
        ]

    @property
    def lemmas(self) -> List[str]:
        """
        Returns list of lowercase lemmas.
        """
        # Lemmatisation of all texts take a lot of time,
        # once calculated save result to bin file.
        saved_lemmas = get_saved_lemmas_for_text(self.text)
        if saved_lemmas:
            return saved_lemmas

        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in self.tokens]

        save_lemmas_for_text(self.text, lemmas)

        return lemmas

    @property
    def sentiment(self) -> Sentiment:
        sentiments = [get_sentiment(sentence) for sentence in self.sentences]

        return Sentiment(min=min(sentiments), avg=sum(sentiments) / len(sentiments), max=max(sentiments))

    @property
    def collocations(self):
        raise NotImplementedError


@dataclass
class News:
    title: Text
    content: Text
    subject: str
    is_fake: bool
    created_on: date

    @property
    def is_true(self) -> bool:
        return not self.is_fake

    @property
    def all_text(self) -> Text:
        return self.title + self.content


def clear_text(text: str) -> str:
    # Remove links
    text = re.sub(r"http\S+", '', text)
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove square brackets
    text = re.sub(r"\[[^]]*\]", '', text)

    text = text.replace("U.S.", "United States")

    return text


def get_saved_lemmas_for_text(text: str) -> List[str]:
    src_path = Path(__file__).parent
    pickle_path = (src_path / "../data/lemmas.bin").resolve()

    if path.exists(pickle_path):
        return pickle.load(open(pickle_path, "rb")).get(text, [])

    return []


def save_lemmas_for_text(text: str, lemmas: List[str]) -> None:
    src_path = Path(__file__).parent
    pickle_path = (src_path / "../data/lemmas.bin").resolve()

    if path.exists(pickle_path):
        all_lemmas = pickle.load(open(pickle_path, "rb"))
    else:
        all_lemmas = {}

    all_lemmas[text] = lemmas

    pickle.dump(all_lemmas, open(pickle_path, "wb"))
