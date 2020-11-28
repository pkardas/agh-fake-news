import logging
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.news import News, count_misspellings, news_length

logger = logging.getLogger()


class AttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, x: List[News]) -> np.array:
        logger.info(f"{len(x)} news to transform")

        result = np.c_[
            count_misspellings(x),
            news_length(x),
        ]

        logger.info("Finished extracting features")

        return result


pipeline = Pipeline([
    ("attrs_adder", AttributesAdder()),
])
