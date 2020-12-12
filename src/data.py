import csv
import logging
import pickle
import re
from datetime import date, datetime
from os import path, walk
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from gensim.models import Word2Vec, Phrases

from src.news import News, Text
from src.tweets import Tweet, TweetType
from src.utils import cached

logger = logging.getLogger()

UniqueId = int
TweetId = str
Delay = float

SourceTweet = Tuple[UniqueId, TweetId, Delay]
DestinationTweet = Tuple[UniqueId, TweetId, Delay]


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


def get_tweets() -> List[Tweet]:
    src_path = Path(__file__).parent

    tweets = []

    pickle_path = (src_path / "../data/tweets.bin").resolve()
    if path.exists(pickle_path):
        logger.info("Tweets available as binary file, loading...")
        return pickle.load(open(pickle_path, "rb"))

    _, _, file_names = next(walk((src_path / "../data/rumor_detection/twitter16/tree").resolve()))
    for file_name in file_names:
        file_path = (src_path / f"../data/rumor_detection/twitter16/tree/{file_name}").resolve()

        logger.info(f"Processing {file_path}")

        with open(file_path, 'r') as file:
            tweet_history = _get_tweet_history(file.read().splitlines())

        tweets.append(_build_tweet(tweet_history))

    pickle.dump(tweets, open(pickle_path, "wb"))

    return tweets


def get_news_labels(all_news: List[News]) -> np.array:
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
            created_on=_format_date(row["date"])
        )
        for i, row in enumerate(data)
        if _format_date(row["date"])  # Analysis showed that there are 10 articles without date
    ]


def _format_date(date_str: str) -> Optional[date]:
    for fmt in ("%B %d, %Y", "%y-%b-%d", "%b %d, %Y"):
        try:
            if fmt == "%B %d, %Y":
                date_str = _add_leading_zero(date_str)

            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass
    return None


def _add_leading_zero(date_str: str) -> str:
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


def _build_tweet(tweet_history: List[Tuple[SourceTweet, DestinationTweet]]) -> Tweet:
    _, root_tuple = tweet_history[0]

    root_tweet = Tweet(
        unique_id=root_tuple[0],
        delay=root_tuple[2],
        content=_get_tweet_text(root_tuple[1]),
        tweet_type=_get_tweet_type(root_tuple[1]),
        children=[]
    )

    for history in tweet_history[1:]:
        source, destination = history

        root_tweet.insert_children(source[0], Tweet(
            unique_id=destination[0],
            delay=destination[2],
            content=_get_tweet_text(destination[1]),
            tweet_type=_get_tweet_type(destination[1]),
            children=[]
        ))

    return root_tweet


@cached(minutes=15, key=lambda tweet_id: tweet_id)
def _get_tweet_text(tweet_id: str) -> str:
    source_tweets_path = (Path(__file__).parent / "../data/rumor_detection/twitter16/source_tweets.txt").resolve()

    with open(source_tweets_path, 'r') as file:
        source_tweets = file.read().splitlines()
        source_tweets = [line.split('	', 1) for line in source_tweets]

    return next((tweet_text for source_tweet_id, tweet_text in source_tweets if source_tweet_id == tweet_id), None)


@cached(minutes=15, key=lambda tweet_id: tweet_id)
def _get_tweet_type(tweet_id: str) -> TweetType:
    labels_path = (Path(__file__).parent / "../data/rumor_detection/twitter16/label.txt").resolve()

    with open(labels_path, 'r') as file:
        labels = file.read().splitlines()
        labels = [line.split(':') for line in labels]

    tweet_type = next((tweet_type for tweet_type, label_tweet_id in labels if label_tweet_id == tweet_id), None)
    return TweetType(tweet_type) if tweet_type else TweetType.UNVERIFIED


def _get_tweet_history(retweets_history: List[str]) -> List[Tuple[SourceTweet, DestinationTweet]]:
    result = []

    for line in retweets_history:
        source, destination = line.replace('[', '').replace(']', '').replace("'", '').split("->")

        source = source.split(", ")
        destination = destination.split(", ")

        source = (int(source[0]) if source[0] != "ROOT" else -1, source[1], float(source[2]))
        destination = (int(destination[0]), destination[1], float(destination[2]))

        result.append((source, destination))

    return result
