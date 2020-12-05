import csv
import logging
import pickle
import re
from datetime import date, datetime
from os import path
from pathlib import Path
from typing import List, Optional

import numpy as np
from gensim.models import Word2Vec, Phrases

from src.news import News, Text

logger = logging.getLogger()


def get_news() -> List[News]:
    logger.info("Fetching news...")

    news = []

    src_path = Path(__file__).parent

    pickle_path = (src_path / "../data/news.bin").resolve()
    if path.exists(pickle_path):
        logger.info("News available as binary file, loading...")
        return pickle.load(open(pickle_path, "rb"))

    true_path = (src_path / "../data/fake_and_real_news/True.csv").resolve()
    fake_path = (src_path / "../data/fake_and_real_news/Fake.csv").resolve()

    with open(true_path, 'r') as file:
        news += _extract_news_from_csv(file, False)

    with open(fake_path, 'r') as file:
        news += _extract_news_from_csv(file, True)

    logger.info("Finished fetching news")

    pickle.dump(news, open(pickle_path, "wb"))

    return news


def get_labels(all_news: List[News]) -> np.array:
    return np.array([int(news.is_fake) for news in all_news])


def get_word_to_vec_model(all_news: List[News]) -> Word2Vec:
    model_path = str((Path(__file__).parent / "../data/word2vec.model").resolve())  # gensim expects path as string

    if path.exists(model_path):
        logger.info("Word2Vec model available...")
        return Word2Vec.load(model_path)

    logger.info("Preparing Word2Vec model...")

    unigrams = [news.all_text.tokens for news in all_news]
    bigrams = Phrases(unigrams)

    model = Word2Vec(bigrams[unigrams], min_count=1, size=64, workers=4, window=4, sg=1)
    model.save(model_path)

    return model


def _extract_news_from_csv(file, is_fake):
    data = csv.DictReader(file)

    return [
        News(
            news_id=i,
            title=Text(f"title-{i}", row["title"]),
            content=Text(f"content-{i}", row["text"]),
            subject=row["subject"],
            is_fake=is_fake,
            created_on=format_date(row["date"])
        )
        for i, row in enumerate(data)
        if format_date(row["date"])  # Analysis showed that there are 10 articles without date
    ]


def format_date(date_str: str) -> Optional[date]:
    for fmt in ("%B %d, %Y", "%y-%b-%d", "%b %d, %Y"):
        try:
            if fmt == "%B %d, %Y":
                date_str = add_leading_zero(date_str)

            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass
    return None


def add_leading_zero(date_str: str) -> str:
    pattern = re.compile(r"[a-zA-Z]+ ([0-9]+), [0-9]+")

    if date_str[-1] == ' ':
        date_str = date_str[:-1]

    found_items = pattern.findall(date_str)

    if not found_items:
        return date_str

    day_of_month = int(found_items[0])

    if day_of_month >= 10:
        return date_str

    return date_str.replace(' ', " 0", 1)
