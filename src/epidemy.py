import logging

from src.data import get_tweets
from src.tweets import plot_all_tweets_propagation

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

tweets = get_tweets()

# true_tweets = [tweet for tweet in tweets if tweet.tweet_type is TweetType.TRUE]
# false_tweets = [tweet for tweet in tweets if tweet.tweet_type is TweetType.FALSE]
# non_rumor_tweets = [tweet for tweet in tweets if tweet.tweet_type is TweetType.NON_RUMOR]
# unverified_tweets = [tweet for tweet in tweets if tweet.tweet_type is TweetType.UNVERIFIED]
#
# plot_single_tweet_propagation(true_tweets[0])
# plot_single_tweet_propagation(false_tweets[0])

plot_all_tweets_propagation(tweets)
