import logging
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.data import get_word_to_vec_model
from src.news import News, english_dictionary
from src.utils import return_saved_data

logger = logging.getLogger()


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


@return_saved_data("news_sentiment")
def news_sentiment(all_news: List[News]) -> np.array:
    """
    For every news returns 3 attributes - min, avg and max sentiment.
    """
    return np.array([
        news.all_text.sentiment.as_list()
        for news in all_news
    ])


@return_saved_data("top_frequent_bigrams")
def top_frequent_bigrams(all_news: List[News]) -> np.array:
    """
    Returns top 10 most frequent bigrams for all news.
    """
    model = get_word_to_vec_model(all_news)

    def bigram_to_vec(bigram: str) -> Optional[np.array]:
        return model.wv[bigram] if bigram in model.wv else None

    bigrams = []

    for news in all_news:
        news_vec_bigrams = []
        news_bigrams = news.all_text.bigrams

        for news_bigram in news_bigrams:
            bigram_as_vec = bigram_to_vec(news_bigram)
            if bigram_as_vec is None:
                continue
            news_vec_bigrams.append(bigram_as_vec)

            # Only 10 bigrams
            if len(news_vec_bigrams) == 10:
                break

        # If news has less than 10 bigrams, add vectors with zeros.
        # Vectors have 64 cells.
        for _ in range(10 - len(news_vec_bigrams)):
            news_vec_bigrams.append(np.zeros(64))

        # Join arrays
        bigrams.append(np.concatenate(tuple(news_vec_bigrams)))

    return np.array(bigrams)


class AttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, x: List[News]) -> np.array:
        logger.info(f"{len(x)} news to transform")

        result = np.c_[
            count_misspellings(x),
            news_length(x),
            news_sentiment(x),
            top_frequent_bigrams(x)
        ]

        logger.info("Finished extracting features")

        return result


pipeline = Pipeline([
    ("attrs_adder", AttributesAdder()),
])


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
