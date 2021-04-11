import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

if __name__ == '__main__':
    # dataset = pd.read_csv('NLP_data.csv')
    # # resource has only 1 associated course
    # subset1 = dataset.groupby(["COMBINED_TITLE"]).count().sort_values(["COURSENAME"], ascending=False).reset_index()[
    #     ['COMBINED_TITLE', 'COURSENAME']]
    # ss1 = list(set(subset1[subset1['COURSENAME'] < 2]['COMBINED_TITLE']))
    # subset_based_on_title = dataset[dataset['COMBINED_TITLE'].isin(ss1)]
    # subset_based_on_title.to_csv("unique_NLP_resources.csv", index=False, encoding='utf-8-sig')
    # print(len(subset_based_on_title))
    data = pd.read_csv('unique_resources.csv')
    processed_features = data.iloc[:, 3].values
    labels = data.iloc[:, 0].values
    struct = namedtuple('struct', 'accuracies n')
    structs = []
    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

    for n in range(1, 4):
        print('\nn: {}'.format(n))
        start_time = time.time()
        vectorizer = CountVectorizer(ngram_range=(n, n), min_df=1)
        vectorizer.fit(X_train)
        fit_time = time.time() - start_time
        x_test = vectorizer.transform(X_test)
        x_train = vectorizer.transform(X_train)

        # gnb = GaussianNB()
        # start_time = time.time()
        # gnb.fit(x_train.toarray(), y_train)
        # print('Fit Time: {}'.format(time.time() - start_time))
        # start_time = time.time()
        # y_pred = gnb.predict(x_test.toarray())
        # print('Pred time: {}'.format(time.time() - start_time))
        # accuracy = metrics.accuracy_score(y_test, y_pred)
        # print('Accuracy: {}'.format(accuracy))
        # print('='*200)
        # s = struct(accuracies=accuracy, n=n)
        # structs.append(s)

        # scoring = ['accuracy']
        # clf = SVC(kernel='linear')
        # scores = cross_validate(clf, x_test, y_test, scoring=scoring, cv=5, return_train_score=False)
        # print(scores)
        # s = struct(accuracies=scores.get('test_accuracy'), n=n)
        # structs.append(s)

        k_range = range(1, 25)
        best_accuracy = [0, 0]
        accuracies = []
        run_times = []
        for k in k_range:
            start_time = time.time()
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            predictions = knn.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
            run_time = time.time() - start_time
            run_times.append(run_time)
            if accuracy > best_accuracy[0]:
                best_accuracy[0] = accuracy
                best_accuracy[1] = k
            print('-' * 200)
            print('K: {}, Accuracy: {}, Fit Time: {}, Run Time: {}'.format(k, accuracy, fit_time, run_time))
        print('=' * 200)
        print('Best Accuracy: {} @ k: {}'.format(best_accuracy[0], best_accuracy[1]))
        print('Average run time: {}'.format(np.mean(run_times)))
        s = struct(accuracies=accuracies, n=n)
        structs.append(s)

    dat = []
    for s in structs:
        dat.append(s.accuracies)

    # # fig = plt.figure()
    # # ax = plt.axes()
    # # ax.plot(dat)
    # #
    fig7, ax7 = plt.subplots()
    ax7.set_title('Ngram Accuracy')
    plt.xlabel('n')
    plt.ylabel('accuracy')
    ax7.boxplot(dat)

    plt.show()

    #
