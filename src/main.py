from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.attributes import pipeline
from src.data import get_news, get_labels
from src.utils import setup


def log_reg(x_train, x_test, y_train, y_test):
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    score = roc_auc_score(y_test, y_pred), f1_score(y_test, y_pred)

    print(score)


def pas_aggr(x_train, x_test, y_train, y_test):
    classifier = PassiveAggressiveClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    score = roc_auc_score(y_test, y_pred), f1_score(y_test, y_pred)

    print(score)


def sdg(x_train, x_test, y_train, y_test):
    classifier = SGDClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    score = roc_auc_score(y_test, y_pred), f1_score(y_test, y_pred)

    print(score)


def random_forrest(x_train, x_test, y_train, y_test):
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    score = roc_auc_score(y_test, y_pred), f1_score(y_test, y_pred)

    print(score)


def svc(x_train, x_test, y_train, y_test):
    classifier = SVC()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    score = roc_auc_score(y_test, y_pred), f1_score(y_test, y_pred)

    print(score)


@setup
def main():
    all_news = get_news()
    labels = get_labels(all_news)

    transformed_news = pipeline.fit_transform(all_news)

    x_train, x_test, y_train, y_test = train_test_split(transformed_news, labels, shuffle=True, random_state=42)

    # Test various classifiers
    log_reg(x_train, x_test, y_train, y_test)
    pas_aggr(x_train, x_test, y_train, y_test)
    sdg(x_train, x_test, y_train, y_test)
    random_forrest(x_train, x_test, y_train, y_test)
    svc(x_train, x_test, y_train, y_test)


main()
