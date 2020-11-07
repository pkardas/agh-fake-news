from src.config import setup
from src.data import get_news
from src.text import get_tokens, get_tokens_without_punctuation, get_tokens_without_stop_words

setup()
data = get_news()
print(len(data))

text = data[0].body
print(text)

tokens = get_tokens(text)
tokens = get_tokens_without_punctuation(tokens)
tokens = get_tokens_without_stop_words(tokens)

print(tokens)
