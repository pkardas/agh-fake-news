import csv
import re
from datetime import date, datetime
from typing import List, Optional

from src.news import News, Text


def get_news() -> List[News]:
    news = []

    with open("data/fake_and_real_news/True.csv", 'r') as file:
        news += _extract_news_from_csv(file, False)

    with open("data/fake_and_real_news/Fake.csv", 'r') as file:
        news += _extract_news_from_csv(file, True)

    return news


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
