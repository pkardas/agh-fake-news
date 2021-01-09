import logging

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, KFold
from tabulate import tabulate

from src.attributes import pipeline
from src.data import get_news, get_news_labels
from src.utils import setup, SELECTED_DATASET
import numpy as np

logger = logging.getLogger()


def check_classifier(classifier, x, y, k_fold=False):
    if k_fold:
        _check_classifier_k_fold(classifier, x, y)
        return

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=42)

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


def _check_classifier_k_fold(classifier, x, y):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    recalls = []
    accuracies = []
    precisions = []
    f1s = []
    roc_auc_scores = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        recalls.append(recall_score(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        roc_auc_scores.append(roc_auc_score(y_test, y_pred))

    columns = ["AVG Recall", "AVG Accuracy", "AVG Precision", "AVG F1", "AVG ROC AUC"]
    data = [
        np.round(np.mean(scores), 3) for scores in [
            recalls, accuracies, precisions, f1s, roc_auc_scores
        ]
    ]

    print(classifier.__class__.__name__)  # Classifier name should not be extracted like this :P
    print(tabulate(headers=columns, tabular_data=[data]))
    print()


@setup
def main():
    all_news = get_news(SELECTED_DATASET)
    labels = get_news_labels(all_news)

    transformed_news = pipeline.fit_transform(all_news)

    # Test various classifiers
    check_classifier(LogisticRegression(max_iter=1000), transformed_news, labels, k_fold=True)
    check_classifier(RandomForestClassifier(), transformed_news, labels, k_fold=True)
    check_classifier(GradientBoostingClassifier(), transformed_news, labels, k_fold=True)
    check_classifier(ExtraTreesClassifier(), transformed_news, labels, k_fold=True)

    check_classifier(VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier()),
            ("gd", GradientBoostingClassifier()),
            ("et", ExtraTreesClassifier())
        ],
        voting="hard"
    ), transformed_news, labels, k_fold=True)

    check_classifier(VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier()),
            ("gd", GradientBoostingClassifier()),
            ("et", ExtraTreesClassifier())
        ],
        voting="soft"
    ), transformed_news, labels, k_fold=True)


main()
