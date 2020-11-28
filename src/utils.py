import logging
import pickle
from functools import wraps
from os import path
from pathlib import Path

import nltk
import numpy as np

logger = logging.getLogger()


def return_saved_data(file_name: str) -> np.array:
    """
    Returns 'data/file_name.bin' if available.
    Otherwise calls 'feature_extractor' and saves data locally.
    """
    def decorator(feature_extractor):
        @wraps(feature_extractor)
        def decorated(all_news):
            pickle_path = (Path(__file__).parent / f"../data/{file_name}.bin").resolve()

            if path.exists(pickle_path):
                logger.info(f"'{file_name}' available")
                return pickle.load(open(pickle_path, "rb"))

            features = feature_extractor(all_news)

            pickle.dump(features, open(pickle_path, "wb"))

            logger.info(f"Saved {len(all_news)} '{file_name}'")

            return features

        return decorated
    return decorator


def setup(main):
    def wrapper():
        # Setup
        nltk.download("vader_lexicon")
        logging.basicConfig(level=logging.INFO)
        _load_lemma()

        main()

        # Teardown
        _save_lemma()

    return wrapper


all_lemmas = {}


def _load_lemma():
    global all_lemmas

    pickle_path = (Path(__file__).parent / "../data/lemmas.bin").resolve()

    if path.exists(pickle_path):
        all_lemmas = pickle.load(open(pickle_path, "rb"))
        logger.info("Lemma loaded")
    else:
        all_lemmas = {}
        logger.info("Lemma unavailable")


def _save_lemma():
    global all_lemmas

    pickle_path = (Path(__file__).parent / "../data/lemmas.bin").resolve()

    pickle.dump(all_lemmas, open(pickle_path, "wb"))

    logger.info("Lemma saved")
