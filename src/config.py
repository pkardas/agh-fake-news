import nltk

import logging

from src.news import load_lemma, save_lemma


# TODO: Move to decorator
def setup():
    nltk.download("vader_lexicon")
    logging.basicConfig(level=logging.INFO)
    load_lemma()


def teardown():
    save_lemma()
