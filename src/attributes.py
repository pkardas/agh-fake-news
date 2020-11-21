import logging
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from src.news import News

logger = logging.getLogger()


class AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_sentiment=False) -> None:
        self.add_sentiment = add_sentiment

    def fit(self, x, y=None):
        return self

    def transform(self, x: List[News]) -> np.array:
        logger.info("Computing lemmas...")

        documents = [' '.join(news.all_text.lemmas) for news in x]

        logger.info("Lemmatisation done, extracting frequency...")

        # Counts occurrence of tokens:
        token_freq_matrix = CountVectorizer().fit_transform(documents)
        # Term Frequency-Inverse Document Frequency:
        tf_idf_matrix = TfidfTransformer().fit_transform(token_freq_matrix)

        if self.add_sentiment:
            # TODO: Add sentiment to the matrix
            pass

        logger.info("Finished extracting features")

        return tf_idf_matrix


pipeline = Pipeline([
    ("attrs_adder", AttributesAdder()),
])
