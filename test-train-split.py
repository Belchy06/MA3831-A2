import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

if __name__ == '__main__':
    data = pd.read_csv('unique_resources.csv')
    processed_features = data.iloc[:, 3].values
    labels = data.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

    for n in range(1, 4):
        print('\n{}'.format(n))
        start_time = time.time()
        vectorizer = CountVectorizer(ngram_range=(1, n), min_df=1)
        vectorizer.fit(X_train)
        fit_time = time.time() - start_time
        x_test = vectorizer.transform(X_test)
        x_train = vectorizer.transform(X_train)

        # k_range = range(1, 25)
        # best_accuracy = [0, 0]
        # run_times = []
        # for k in k_range:
        #     start_time = time.time()
        #     knn = KNeighborsClassifier(n_neighbors=k)
        #     knn.fit(x_train, y_train)
        #     predictions = knn.predict(x_test)
        #     accuracy = accuracy_score(y_test, predictions)
        #     run_time = time.time() - start_time
        #     run_times.append(run_time)
        #     if accuracy > best_accuracy[0]:
        #         best_accuracy[0] = accuracy
        #         best_accuracy[1] = k
        #     print('-' * 200)
        #     print('K: {}, Accuracy: {}, Fit Time: {}, Run Time: {}'.format(k, accuracy, fit_time, run_time))
        # print('=' * 200)
        # print('Best Accuracy: {} @ k: {}'.format(best_accuracy[0], best_accuracy[1]))
        # print('Average run time: {}'.format(np.mean(run_times)))

        scoring = ['accuracy']
        clf = SVC(kernel='linear')
        scores = cross_validate(clf, x_test, y_test, scoring=scoring, cv=5, return_train_score=False)
        print(scores)

    # scoring = ['accuracy']
    # clf = SVC(kernel='linear')
    # scores = cross_validate(clf, x_test, scoring=scoring, cv=5, return_train_score=False)
    # print(scores)
    # print('X: {}\nY: {}'.format(y_train, y_test))
    # k_range = range(1, 25)
    # best_accuracy = [0, 0]
    # run_times = []
    # for k in k_range:
    #     start_time = time.time()
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train, y_train)
    #     predictions = knn.predict(X_test)
    #     # print(confusion_matrix(y_test, predictions))
    #     # print(classification_report(y_test, predictions))
    #     accuracy = accuracy_score(y_test, predictions)
    #     run_time = time.time() - start_time
    #     run_times.append(run_time)
    #     if accuracy > best_accuracy[0]:
    #         best_accuracy[0] = accuracy
    #         best_accuracy[1] = k
    #     print('-' * 200)
    #     print('K: {}'.format(k))
    #     print('Accuracy: {}'.format(accuracy))
    #     print('Run time: {}'.format(run_time))
    # print('=' * 200)
    # print('Best Accuracy: {} @ k: {}'.format(best_accuracy[0], best_accuracy[1]))
    # print('Average run time: {}'.format(np.mean(run_times)))
