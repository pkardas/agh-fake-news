import re
from dataclasses import dataclass
from datetime import date
from typing import List

from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

from src.features import get_sentiment
from src.models import Sentiment


class Text:
    def __init__(self, text: str):
        self.text = clear_text(text)

    def __str__(self) -> str:
        return self.text

    @property
    def sentences(self) -> List[str]:
        return sent_tokenize(self.text)

    @property
    def tokens(self) -> List[str]:
        """
        Returns list of lowercase tokens without stop words.
        """
        tokens = word_tokenize(self.text)
        return [
            token.lower()
            for token in tokens
            if token.isalpha() and token.lower() not in stopwords.words("english")
        ]

    @property
    def lemmas(self) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in self.tokens]

    @property
    def sentiment(self) -> Sentiment:
        sentiments = [get_sentiment(sentence) for sentence in self.sentences]

        return Sentiment(min=min(sentiments), avg=sum(sentiments) / len(sentiments), max=max(sentiments))

    @property
    def collocations(self):
        raise NotImplemented


@dataclass
class News:
    title: Text
    content: Text
    subject: str
    is_fake: bool
    created_on: date

    @property
    def is_true(self) -> bool:
        return not self.is_fake


def clear_text(text: str) -> str:
    # Remove links
    text = re.sub(r"http\S+", '', text)
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove square brackets
    text = re.sub(r"\[[^]]*\]", '', text)

    text = text.replace("U.S.", "United States")

    return text
