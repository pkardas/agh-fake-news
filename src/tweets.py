from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx
import numpy as np
from networkx import Graph
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
    parent_unique_id: Optional[int] = None

    def insert_children(self, parent_unique_id: int, children: Tweet) -> bool:
        if self.unique_id == parent_unique_id:
            children.parent_unique_id = parent_unique_id
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

    def get_reproduction_num(self, include_non_spreaders: bool = True) -> float:
        root_spreaders = defaultdict(list)

        for tweet in self.all_children:
            root_spreaders[tweet.parent_unique_id].append(tweet)

        num_of_non_spreaders = len(self.all_children) + 1 - len(root_spreaders)

        return np.average(
            [len(infected) for infected in root_spreaders.values()] +
            [0] * (num_of_non_spreaders if include_non_spreaders else 0)
        )

    def get_single_propagation_scatter(self, color: Optional[str] = None) -> Scatter:
        time = defaultdict(int)

        for child in self.all_children:
            time[child.delay] += 1

        x = sorted(time.keys())
        y = []

        for delay in x:
            prev_sum = y[-1] if y else 0
            y.append(prev_sum + time[delay])

        return Scatter(x=x, y=y, marker=Marker(color=color) if color else None, text=self.content)

    def plot_propagation(self) -> None:
        Figure(
            data=[self.get_single_propagation_scatter()],
            layout=Layout(
                template="plotly_white",
                title=Title(text=f"Tweets propagation, root: {self.unique_id}, {self.tweet_type}", x=0.5),
                yaxis=YAxis(title="Cumulative number of retweets"),
                xaxis=XAxis(title="Delay since root tweet"),
            )
        ).show()

    def plot_relations(self) -> None:
        tree = self._get_graph()

        pos = networkx.spring_layout(tree)

        plt.figure(figsize=(16, 16))
        plt.title(f"{self.tweet_type}: '{self.content}'")

        networkx.draw_networkx_nodes(
            tree, pos,
            nodelist=tree.nodes,
            node_color='b',
            node_size=100,
            alpha=0.8
        )
        networkx.draw_networkx_nodes(
            tree, pos,
            nodelist=[self.unique_id],
            node_color='g' if self.tweet_type is TweetType.TRUE else 'r',
            node_size=500,
            alpha=0.8
        )
        networkx.draw_networkx_edges(tree, pos, width=1.0, alpha=0.5)

        plt.show()

    def _get_graph(self) -> Graph:
        g = Graph()

        for current_tweet in [self] + self.all_children:
            if not current_tweet.parent_unique_id:
                g.add_node(current_tweet.unique_id)  # root
                continue
            g.add_edge(current_tweet.parent_unique_id, current_tweet.unique_id)

        return g


def plot_all_tweets_propagation(all_tweets: List[Tweet]) -> None:
    colors = {
        TweetType.TRUE: "green",
        TweetType.FALSE: "red"
    }

    all_tweets.sort(key=lambda tweet: tweet.tweet_type)

    Figure(
        data=[
            tweet.get_single_propagation_scatter(colors[tweet.tweet_type])
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
