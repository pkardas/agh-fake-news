import logging

from src.data import get_tweets

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

tweets = get_tweets()

print(tweets)
