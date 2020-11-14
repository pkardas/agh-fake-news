from nltk.sentiment import SentimentIntensityAnalyzer


def get_sentiment(sentence: str) -> float:
    """
    Returns sentiment of a sentence.
      sentiment > 0 - positive feelings,
      sentiment ~ 0 - neutral feelings,
      sentiment < 0 - negative feelings.

    Shows bias in both directions.
    """
    return SentimentIntensityAnalyzer().polarity_scores(sentence)["compound"]


def get_unusual_words_ratio():
    """
    Words not appearing in the English dictionary.
    """
    raise NotImplemented


def get_nouns_per_sentence_ratio():
    raise NotImplemented


def get_adjs_per_sentence_ratio():
    raise NotImplemented
