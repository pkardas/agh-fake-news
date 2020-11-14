from datetime import date
from typing import List, Optional

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from src.features import get_sentiment
from src.models import Sentiment


class Text:
    def __init__(self, text: str):
        self.text = text

    @property
    def sentences(self) -> List[str]:
        return sent_tokenize(self.text)

    @property
    def tokens(self) -> List[str]:
        tokens = word_tokenize(self.text)
        return [
            token.lower()
            for token in tokens
            if token.isalpha() and token.lower() not in stopwords.words("english")
        ]

    @property
    def sentiment(self) -> Sentiment:
        sentiments = [get_sentiment(sentence) for sentence in self.sentences]

        return Sentiment(min=min(sentiments), avg=sum(sentiments) / len(sentiments), max=max(sentiments))


class News:
    def __init__(self, title: Text, content: Text, subject: str, is_fake: bool, created_on: Optional[date]):
        self.title = title
        self.content = content
        self.is_fake = is_fake
        self.subject = subject
        self.created_at = created_on

    @property
    def is_true(self) -> bool:
        return not self.is_fake
