from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class TweetMetrics(BaseModel):
    retweet_count: int
    reply_count: int
    like_count: int
    quote_count: int


class ReferencedTweet(BaseModel):
    type: str
    id: str


class TwitterMention(BaseModel):
    start: int
    end: str
    username: str


class TwitterUrl(BaseModel):
    start: int
    end: str
    url: str


class TwitterEntities(BaseModel):
    mentions: List[TwitterMention] = []
    urls: List[TwitterUrl] = []


class TwitterContextAnnotationDomain(BaseModel):
    id: str
    name: str
    description: str


class TwitterContextAnnotationEntity(BaseModel):
    id: str
    name: str


class TwitterContextAnnotation(BaseModel):
    domain: TwitterContextAnnotationDomain
    entity: TwitterContextAnnotationEntity


class TwitterGeo(BaseModel):
    place_id: str


class Tweet(BaseModel):
    id: str
    author_id: str
    text: str
    lang: str
    possibly_sensitive: bool
    created_at: datetime
    public_metrics: TweetMetrics
    referenced_tweets: List[ReferencedTweet] = []
    context_annotations: List[TwitterContextAnnotation] = []
    geo: Optional[TwitterGeo]
    entities: Optional[TwitterEntities]
    source: str
