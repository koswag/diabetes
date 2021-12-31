import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

Classifier = KNeighborsClassifier | SVC | BaggingClassifier


def evaluate(classifier: Classifier, train_data: pd.DataFrame, test_data: pd.DataFrame):
    classifier = fit(classifier, train_data)
    actual, predicted = validate(classifier, test_data)

    name = classifier.__class__.__name__
    print(f'{name} result'.center(53, '='), end = '\n\n')
    print(classification_report(actual, predicted), end = '\n\n')
    return classifier


def fit(classifier: Classifier, train_data: pd.DataFrame):
    *features, label = train_data.columns

    return classifier.fit(
        X = train_data[features],
        y = train_data[label]
    )


def validate(classifier: KNeighborsClassifier, test_data: pd.DataFrame):
    *features, label = test_data.columns

    actual = test_data[label]
    predicted = classifier.predict(X = test_data[features])
    return actual, predicted
