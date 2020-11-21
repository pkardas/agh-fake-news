from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.attributes import pipeline
from src.config import setup, teardown
from src.data import get_news, get_labels

setup()

all_news = get_news()
labels = get_labels(all_news)

transformed_news = pipeline.fit_transform(all_news)

x_train, x_test, y_train, y_test = train_test_split(transformed_news, labels, shuffle=True, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

print(log_reg.score(x_test, y_test) * 100)

teardown()
