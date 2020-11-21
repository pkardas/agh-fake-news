import csv
import logging
import pickle
import re
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from os import path
import numpy as np

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


def _extract_news_from_csv(file, is_fake):
    data = csv.DictReader(file)

    return [
        News(
            title=Text(row["title"]),
            content=Text(row["text"]),
            subject=row["subject"],
            is_fake=is_fake,
            created_on=format_date(row["date"])
        )
        for row in data
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
