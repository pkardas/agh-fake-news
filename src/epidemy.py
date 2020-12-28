import logging

import numpy as np
from scipy.stats import hmean
from tabulate import tabulate

from src.data import get_tweets
from src.tweets import TweetType

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

tweets = get_tweets(use_local=True)

true_tweets = [tweet for tweet in tweets if tweet.tweet_type is TweetType.TRUE]
false_tweets = [tweet for tweet in tweets if tweet.tweet_type is TweetType.FALSE]

# plot_single_tweet_propagation(true_tweets[0])
# plot_single_tweet_propagation(false_tweets[0])

# plot_all_tweets_propagation(tweets)
# false_tweets[1].plot_relations()
# true_tweets[2].plot_relations()

true_repr_nums = [true_tweet.get_reproduction_num(False) for true_tweet in true_tweets]
true_repr_nums_all = [true_tweet.get_reproduction_num(True) for true_tweet in true_tweets]

false_repr_nums = [false_tweet.get_reproduction_num(False) for false_tweet in false_tweets]
false_repr_nums_all = [false_tweet.get_reproduction_num(True) for false_tweet in false_tweets]

columns = [
    "True/False",
    "only_spreaders: Mean", "only_spreaders: HMean", "only_spreaders: Min", "only_spreaders: Max"
]

table_data = [
    [
        "true_tweets",
        np.average(true_repr_nums), hmean(true_repr_nums), np.min(true_repr_nums), np.max(true_repr_nums),
    ],
    [
        "false_tweets",
        np.average(false_repr_nums), hmean(false_repr_nums), np.min(false_repr_nums), np.max(false_repr_nums),
    ],
]
print(tabulate(table_data, headers=columns))

columns = [
    "True/False",
    "all: Mean", "all: HMean", "all: Min", "all: Max"
]

table_data = [
    [
        "true_tweets",
        np.average(true_repr_nums_all), hmean(true_repr_nums_all),
        np.min(true_repr_nums_all), np.max(true_repr_nums_all),
    ],
    [
        "false_tweets",
        np.average(false_repr_nums_all), hmean(false_repr_nums_all),
        np.min(false_repr_nums_all), np.max(false_repr_nums_all),
    ],
]
print(tabulate(table_data, headers=columns))
