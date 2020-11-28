from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import List

import enchant
import numpy as np
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

import src.utils
from src.models import Sentiment
from src.utils import return_saved_data

logger = logging.getLogger()

# enchant is much faster than NLTK when it comes to word lookup (C bindings)
english_dictionary = enchant.Dict("en_US")


class Text:
    def __init__(self, text_id: str, text: str):
        self.text_id = text_id
        self.text = _clear_text(text)

    def __str__(self) -> str:
        return self.text

    def __add__(self, other) -> Text:
        if not isinstance(other, Text):
            return self
        return Text(f"{self.text_id}+{other.text_id}", self.text + ' ' + other.text)

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

        saved_lemmas = src.utils.all_lemmas.get(self.text_id, [])
        if saved_lemmas:
            logger.info(f"Text {self.text_id} lemma available")
            return saved_lemmas

        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in self.tokens]

        src.utils.all_lemmas[self.text_id] = lemmas

        logger.info(f"Text {self.text_id} lemmatised")

        return lemmas

    @property
    def sentiment(self) -> Sentiment:
        """
        Returns sentiment of a sentence.
         sentiment > 0 - positive feelings,
         sentiment ~ 0 - neutral feelings,
         sentiment < 0 - negative feelings.

        Shows bias in both directions.
        """
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiments = [
            sentiment_analyzer.polarity_scores(sentence)["compound"]
            for sentence in self.sentences
        ]

        return Sentiment(min=min(sentiments), avg=sum(sentiments) / len(sentiments), max=max(sentiments))


@dataclass
class News:
    news_id: int
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


@return_saved_data("misspellings")
def count_misspellings(all_news: List[News]) -> np.array:
    """
    Counts words not appearing in the English dictionary.
    """
    return np.array([
        sum(_is_misspelled(word) for word in news.all_text.lemmas)
        for news in all_news
    ])


@return_saved_data("news_lengths")
def news_length(all_news: List[News]) -> np.array:
    return np.array([
        len(news.all_text.lemmas)
        for news in all_news
    ])


def top_frequent_bigrams(all_news: List[News]) -> np.array:
    """
    Returns top 10 most frequent bigrams for all news.
    """
    raise NotImplementedError


def _clear_text(text: str) -> str:
    # Remove links
    text = re.sub(r"http\S+", '', text)
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove square brackets
    text = re.sub(r"\[[^]]*\]", '', text)

    text = text.replace("U.S.", "United States")

    return text


def _is_misspelled(word: str) -> bool:
    # Enchant is case sensitive when it comes to for example names
    capitalized = word.capitalize()
    upper_cased = word.upper()

    is_correct = (
            english_dictionary.check(word) or
            english_dictionary.check(capitalized) or
            english_dictionary.check(upper_cased)
    )
    return not is_correct
