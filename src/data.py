import csv
import os
import pickle
from typing import List

from src.models import News


def get_news() -> List[News]:
    news = []

    if is_news_dump_available():
        return pickle.load(open("../data/news.dump", "rb"))

    with open("../data/fake_and_real_news/True.csv", 'r') as file:
        news += _extract_news_from_csv(file, False)

    with open("../data/fake_and_real_news/Fake.csv", 'r') as file:
        news += _extract_news_from_csv(file, True)

    pickle.dump(news, open("../data/news.dump", "wb"))

    return news


def get_troll_news() -> List[str]:
    news = []

    if is_troll_dump_available():
        return pickle.load(open("../data/troll.dump", "rb"))

    with open("../data/russian_trolls/tweets.csv", 'r') as file:
        data = csv.DictReader(file)

        for row in data:
            news.append(row["text"])

    pickle.dump(news, open("../data/troll.dump", "wb"))

    return news


def is_news_dump_available() -> bool:
    return os.path.exists("../data/news.dump")


def is_troll_dump_available() -> bool:
    return os.path.exists("../data/trolls.dump")


def _extract_news_from_csv(file, is_fake):
    data = csv.DictReader(file)

    return [
        News(is_fake=is_fake, body=row["text"])
        for row in data
    ]
