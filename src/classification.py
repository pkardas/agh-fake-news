import logging

from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from src.attributes import pipeline
from src.data import get_news, get_news_labels
from src.utils import setup

logger = logging.getLogger()


def test_classifier(classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    columns = ["Recall", "Accuracy", "Precision", "F1", "ROC AUC"]
    scores = [
        round(metric(y_test, y_pred), 3) for metric in [
            recall_score, accuracy_score, precision_score, f1_score, roc_auc_score
        ]
    ]

    print(classifier.__class__.__name__)  # Classifier name should not be extracted like this :P
    print(tabulate(headers=columns, tabular_data=[scores]))
    print()


@setup
def main():
    all_news = get_news()
    labels = get_news_labels(all_news)

    transformed_news = pipeline.fit_transform(all_news)

    data = train_test_split(transformed_news, labels, shuffle=True, random_state=42)

    # Test various classifiers
    test_classifier(LogisticRegression(max_iter=1000), *data)
    test_classifier(PassiveAggressiveClassifier(), *data)
    test_classifier(SGDClassifier(), *data)
    test_classifier(RandomForestClassifier(), *data)
    test_classifier(AdaBoostClassifier(), *data)
    test_classifier(GradientBoostingClassifier(), *data)
    test_classifier(ExtraTreesClassifier(), *data)


main()
