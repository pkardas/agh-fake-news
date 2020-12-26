from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import List

import enchant
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer, BigramCollocationFinder, BigramAssocMeasures
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import src.utils

logger = logging.getLogger()

# enchant is much faster than NLTK when it comes to word lookup (C bindings)
english_dictionary = enchant.Dict("en_US")


@dataclass
class Sentiment:
    min: float
    avg: float
    max: float

    def as_list(self) -> List[float]:
        return [self.min, self.avg, self.max]


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
        return get_tokens(self.text)

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

    @property
    def subjectivity(self) -> float:
        """
        Positive 'subjectivity' means text is subjective,
        'subjectivity' close to 0, means text is objective.
        """
        _, subjectivity = TextBlob(self.text).sentiment
        return subjectivity

    @property
    def bigrams(self) -> List[str]:
        """
        Returns list of bigrams, words inside bigrams are space-separated.
        Returns up to 50 bigrams.
        """
        finder = BigramCollocationFinder.from_words(self.tokens)
        return [
            f"{word_0}_{word_1}"
            for word_0, word_1 in finder.nbest(BigramAssocMeasures().pmi, 50)
        ]


@dataclass
class News:
    news_id: str
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


def get_tokens(text: str) -> List[str]:
    tokens = word_tokenize(text)
    return [
        token.lower()
        for token in tokens
        if token.isalpha() and token.lower() not in stopwords.words("english")
    ]


def _clear_text(text: str) -> str:
    # Remove links
    text = re.sub(r"http\S+", '', text)
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove square brackets
    text = re.sub(r"\[[^]]*\]", '', text)

    text = text.replace("U.S.", "United States")

    return text
