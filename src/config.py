import nltk

import logging


def setup():
    nltk.download("vader_lexicon")
    logging.basicConfig(level=logging.INFO)
