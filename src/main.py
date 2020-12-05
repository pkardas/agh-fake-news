from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tabulate import tabulate

from src.attributes import pipeline
from src.data import get_news, get_labels, get_word_to_vec_model
from src.utils import setup


def test_classifier(classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    columns = ["Recall", "Accuracy", "F1", "ROC AUC"]
    scores = [
        round(metric(y_test, y_pred), 3) for metric in [recall_score, accuracy_score, f1_score, roc_auc_score]
    ]

    print(classifier.__class__.__name__)  # Classifier name should not be extracted like this :P
    print(tabulate(headers=columns, tabular_data=[scores]))
    print()


@setup
def main():
    all_news = get_news()
    labels = get_labels(all_news)

    transformed_news = pipeline.fit_transform(all_news)

    data = train_test_split(transformed_news, labels, shuffle=True, random_state=42)

    # Test various classifiers
    test_classifier(LogisticRegression(max_iter=1000), *data)
    test_classifier(PassiveAggressiveClassifier(), *data)
    test_classifier(SGDClassifier(), *data)
    test_classifier(RandomForestClassifier(), *data)
    test_classifier(SVC(), *data)
    test_classifier(GaussianNB(), *data)


main()
