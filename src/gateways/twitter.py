import os

import requests

from src.errors import DependantServiceError
from src.gateways.models import Tweet

ALL_FIELDS = ("?tweet.fields=attachments,author_id,context_annotations,created_at,entities,geo,id,in_reply_to_user_id,"
              "lang,possibly_sensitive,public_metrics,referenced_tweets,source,text,withheld")

HEADERS = {"Authorization": "Bearer {}".format(os.getenv("TWITTER_BEARER_TOKEN"))}


def get_tweet(tweet_id: int) -> Tweet:
    url = f"https://api.twitter.com/2/tweets/{tweet_id}" + ALL_FIELDS
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise DependantServiceError(f"Twitter unreachable. Got <{response.status_code}>.")

    return Tweet(**response.json()["data"])
