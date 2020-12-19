from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from plotly.graph_objs import Scatter, Figure, Layout
from plotly.graph_objs.layout import YAxis, XAxis, Title
from plotly.graph_objs.scatter import Marker


class TweetType(str, Enum):
    TRUE = "true"
    FALSE = "false"
    NON_RUMOR = "non-rumor"
    UNVERIFIED = "unverified"


@dataclass
class Tweet:
    unique_id: int
    delay: float
    content: str
    tweet_type: TweetType
    children: List[Tweet]

    def insert_children(self, parent_unique_id: int, children: Tweet) -> bool:
        if self.unique_id == parent_unique_id:
            self.children.append(children)
            return True

        for child in self.children:
            inserted = child.insert_children(parent_unique_id, children)

            if inserted:
                return True

        return False

    @property
    def all_children(self) -> List[Tweet]:
        all_children = []

        for child in self.children:
            all_children += child.all_children

        return self.children + all_children


def plot_single_tweet_propagation(tweet: Tweet) -> None:
    Figure(
        data=[_get_single_tweet_scatter(tweet)],
        layout=Layout(
            template="plotly_white",
            title=Title(text=f"Tweets propagation, root: {tweet.unique_id}, {tweet.tweet_type}", x=0.5),
            yaxis=YAxis(title="Cumulative number of retweets"),
            xaxis=XAxis(title="Delay since root tweet"),
        )
    ).show()


def plot_all_tweets_propagation(all_tweets: List[Tweet]) -> None:
    colors = {
        TweetType.TRUE: "green",
        TweetType.FALSE: "red"
    }

    all_tweets.sort(key=lambda tweet: tweet.tweet_type)

    Figure(
        data=[
            _get_single_tweet_scatter(tweet, colors[tweet.tweet_type])
            for tweet in all_tweets
            if tweet.tweet_type in {TweetType.TRUE, TweetType.FALSE}
        ],
        layout=Layout(
            template="plotly_white",
            title=Title(text="Tweets propagation", x=0.5),
            yaxis=YAxis(title="Cumulative number of retweets"),
            xaxis=XAxis(title="Delay since root tweet"),
        )
    ).show()


def _get_single_tweet_scatter(tweet: Tweet, color: Optional[str] = None) -> Scatter:
    time = defaultdict(int)

    for child in tweet.all_children:
        time[child.delay] += 1

    x = sorted(time.keys())
    y = []

    for delay in x:
        prev_sum = y[-1] if y else 0
        y.append(prev_sum + time[delay])

    return Scatter(x=x, y=y, marker=Marker(color=color) if color else None, text=tweet.content)
