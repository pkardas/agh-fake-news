from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


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
