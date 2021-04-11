import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    data = pd.read_csv('lemmed_dat2.csv')
    processed_features = data.iloc[:, 3].values
    labels = data.iloc[:, 0].values
    print(processed_features)

    vectorizer = TfidfVectorizer()
    processed_features = vectorizer.fit_transform(processed_features).toarray()

    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
    print('X: {}\nY: {}'.format(y_train, y_test))
    k_range = range(1, 25)
    best_accuracy = [0, 0]
    run_times = []
    for k in k_range:
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        # print(confusion_matrix(y_test, predictions))
        # print(classification_report(y_test, predictions))
        accuracy = accuracy_score(y_test, predictions)
        run_time = time.time() - start_time
        run_times.append(run_time)
        if accuracy > best_accuracy[0]:
            best_accuracy[0] = accuracy
            best_accuracy[1] = k
        print('-' * 200)
        print('K: {}'.format(k))
        print('Accuracy: {}'.format(accuracy))
        print('Run time: {}'.format(run_time))
    print('=' * 200)
    print('Best Accuracy: {} @ k: {}'.format(best_accuracy[0], best_accuracy[1]))
    print('Average run time: {}'.format(np.mean(run_times)))

