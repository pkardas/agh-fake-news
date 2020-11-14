from src.config import setup
from src.data import get_news

setup()

all_news = get_news()

for news in all_news:
    print(news.title)