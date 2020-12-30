import logging
import pickle
from enum import Enum
from functools import wraps
from os import path
from pathlib import Path
from uuid import uuid4

import nltk
from cachelib import SimpleCache

logger = logging.getLogger()


class NewsDataset(str, Enum):
    DATASET_0 = "dataset_0"  # https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset


SELECTED_DATASET = NewsDataset.DATASET_0


def return_saved_data(feature_extractor):
    """
    Returns 'data/file_name.bin' if available.
    Otherwise calls 'feature_extractor' and saves data locally.
    """

    file_name = feature_extractor.__name__

    def wrapper(all_news, dataset):
        pickle_path = (Path(__file__).parent / f"../data/{file_name}-{dataset}.bin").resolve()

        if path.exists(pickle_path):
            logger.info(f"'{file_name}' available")
            return pickle.load(open(pickle_path, "rb"))

        features = feature_extractor(all_news)

        pickle.dump(features, open(pickle_path, "wb"))

        logger.info(f"Saved {len(all_news)} '{file_name}'")

        return features

    return wrapper


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


NO_DATA = f"{uuid4}-NO_DATA"


def cached(*, hours=0, minutes=0, key=lambda x: x):
    """
    Basic caching functionality used for service calls.
     - A decorator, allowing to specify the HOURS (additionally MINUTES) and
       and the key for the cache
    Typical usage:
    ```
    @cached(hours=24, key=lambda x: x)
    def f(x):
        return x*2
    ```
    """
    timeout = hours * 60 * 60 + minutes * 60

    def decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            cache_id = key(*args, **kwargs)
            return_value = decorated.cache.get(cache_id)

            if return_value is None:
                return_value = func(*args, **kwargs)
                decorated.cache.set(cache_id, return_value if return_value is not None else NO_DATA, timeout=timeout)
            elif return_value == NO_DATA:
                return_value = None

            return return_value

        decorated.cache = SimpleCache()
        return decorated

    return decorator


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
