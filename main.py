import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from config import DATA_PATH, SEPARATOR, N_NEIGHBOURS, SVC_KERNEL, TEST_SIZE
from util.prediction import evaluate
from util.preprocessing import preprocess

pd.options.display.max_columns = None


def main():
    raw = pd.read_csv(DATA_PATH, sep = SEPARATOR)
    preprocessed = preprocess(raw)
    train, test = train_test_split(preprocessed, test_size = TEST_SIZE)

    knn = KNeighborsClassifier(n_neighbors = N_NEIGHBOURS)
    evaluate(knn, train, test)

    base = KNeighborsClassifier(n_neighbors = N_NEIGHBOURS)
    ensemble_knn = BaggingClassifier(base, max_samples = 0.5, max_features = 0.5)
    evaluate(ensemble_knn, train, test)

    svc = SVC(kernel = SVC_KERNEL)
    evaluate(svc, train, test)

    linear_svc = LinearSVC()
    evaluate(linear_svc, train, test)


if __name__ == '__main__':
    main()
