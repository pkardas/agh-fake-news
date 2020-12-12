import logging
import pickle
from functools import wraps
from os import path
from pathlib import Path
from uuid import uuid4

import nltk
import numpy as np
from cachelib import SimpleCache

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


NO_DATA = f"{uuid4}-NO_DATA"


def cached(*, hours=0, minutes=0, key=lambda x: x, threshold=None):
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
    if threshold is None:
        if timeout == 0:
            # No expiration was set -- we do not want items to be evicted at all
            # Since we must put a value, let's use something reasonably big
            threshold = 100_000
        elif timeout >= 86400:
            # In most cases, we want caches to last quite some time; in this
            # case the default value is most probably too small to hold all
            # necessary data -- so let's put it 20x higher
            threshold = 10_000
        else:
            # Default from SimpleCache
            threshold = 500

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

        decorated.cache = SimpleCache(threshold=threshold)
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
