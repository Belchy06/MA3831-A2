# # import pandas as pd
# # import time
# #
# # from sklearn.feature_extraction.text import CountVectorizer
# # from sklearn.metrics import accuracy_score
# # from sklearn.model_selection import train_test_split, cross_validate
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn import metrics
# # from collections import namedtuple
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.svm import SVC
# #
# # if __name__ == '__main__':
# #     # dataset = pd.read_csv('NLP_data.csv')
# #     # # resource has only 1 associated course
# #     # subset1 = dataset.groupby(["COMBINED_TITLE"]).count().sort_values(["COURSENAME"], ascending=False).reset_index()[
# #     #     ['COMBINED_TITLE', 'COURSENAME']]
# #     # ss1 = list(set(subset1[subset1['COURSENAME'] < 2]['COMBINED_TITLE']))
# #     # subset_based_on_title = dataset[dataset['COMBINED_TITLE'].isin(ss1)]
# #     # subset_based_on_title.to_csv("unique_NLP_resources.csv", index=False, encoding='utf-8-sig')
# #     # print(len(subset_based_on_title))


# #     data = pd.read_csv('unique_NLP_resources.csv')
# #     processed_features = data.iloc[:, 3]
# #     processed_features = processed_features.fillna(' ')
# #     labels = data.iloc[:, 0]
# #     struct = namedtuple('struct', 'accuracies n')
# #     structs = []
# #     X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
# #
# #     for n in range(1, 4):
# #         print('\nn: {}'.format(n))
# #         start_time = time.time()
# #         vectorizer = CountVectorizer(ngram_range=(n, n), min_df=1)
# #         vectorizer.fit(X_train)
# #         fit_time = time.time() - start_time
# #         x_test = vectorizer.transform(X_test)
# #         x_train = vectorizer.transform(X_train)
# #
# #         gnb = GaussianNB()
# #         start_time = time.time()
# #         gnb.fit(x_train.toarray(), y_train)
# #         print('Fit Time: {}'.format(time.time() - start_time))
# #         print('X test size: {}'.format(x_test.toarray().shape))
# #         start_time = time.time()
# #         y_pred = gnb.predict(x_test.toarray())
# #         print('Pred time: {}'.format(time.time() - start_time))
# #         accuracy = metrics.accuracy_score(y_test, y_pred)
# #         print('Accuracy: {}'.format(accuracy))
# #         print('='*200)
# #         s = struct(accuracies=accuracy, n=n)
# #         structs.append(s)
# #
# #         # scoring = ['accuracy']
# #         # clf = SVC(kernel='linear')
# #         # scores = cross_validate(clf, x_test, y_test, scoring=scoring, cv=5, return_train_score=False)
# #         # print(scores)
# #         # s = struct(accuracies=scores.get('test_accuracy'), n=n)
# #         # structs.append(s)
# #
# #         # k_range = range(1, 25)
# #         # best_accuracy = [0, 0]
# #         # accuracies = []
# #         # run_times = []
# #         # for k in k_range:
# #         #     start_time = time.time()
# #         #     knn = KNeighborsClassifier(n_neighbors=k)
# #         #     knn.fit(x_train, y_train)
# #         #     predictions = knn.predict(x_test)
# #         #     accuracy = accuracy_score(y_test, predictions)
# #         #     accuracies.append(accuracy)
# #         #     run_time = time.time() - start_time
# #         #     run_times.append(run_time)
# #         #     if accuracy > best_accuracy[0]:
# #         #         best_accuracy[0] = accuracy
# #         #         best_accuracy[1] = k
# #         #     print('-' * 200)
# #         #     print('K: {}, Accuracy: {}, Fit Time: {}, Run Time: {}'.format(k, accuracy, fit_time, run_time))
# #         # print('=' * 200)
# #         # print('Best Accuracy: {} @ k: {}'.format(best_accuracy[0], best_accuracy[1]))
# #         # print('Average run time: {}'.format(np.mean(run_times)))
# #         # s = struct(accuracies=accuracies, n=n)
# #         # structs.append(s)
# #
# #     dat = []
# #     for s in structs:
# #         dat.append(s.accuracies)
# #
# #     # # fig = plt.figure()
# #     # # ax = plt.axes()
# #     # # ax.plot(dat)
# #     # #
# #     fig7, ax7 = plt.subplots()
# #     ax7.set_title('Ngram Accuracy')
# #     plt.xlabel('n')
# #     plt.ylabel('accuracy')
# #     ax7.boxplot(dat)
# #
# #     plt.show()
# #
# #     #
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import cufflinks as cf
from sklearn.model_selection import train_test_split

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


data = pd.read_csv('filtered_unique_SW+Lem.csv')
data['SUBJECTS'] = data['SUBJECTS'].values.astype(str)
data['COUNT'] = data['SUBJECTS'].str.split().str.len()
dat = data['COUNT'].values
# mean engagements
print('Mean number of subject tags per resource: {}'.format(dat.mean()))
plt.figure(0)
plt.hist(dat, bins=int(len(data['COUNT'].unique())/2))
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.title('Word Count in Subject Metadata Distribution')
plt.show()
print()

common_words = get_top_n_words(data['SUBJECTS'], 10)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['SUBJECTS' , 'count'])
df1.groupby('SUBJECTS').sum()['count'].sort_values(ascending=False)
plt.figure(1)
plt.bar(df1['SUBJECTS'], df1['count'])
plt.xlabel('Word')
plt.xticks(rotation=45)
# plt.xticks(fontsize=6)
plt.ylabel('Count')
plt.title('Top 10 Words in Subject Tags After Removing Stop Words and Lemmatizing', fontsize=10)
plt.show()

common_words = get_top_n_bigram(data['SUBJECTS'], 10)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['SUBJECTS' , 'count'])
df3.groupby('SUBJECTS').sum()['count'].sort_values(ascending=False)
plt.figure(2)
plt.bar(df3['SUBJECTS'], df3['count'])
plt.xlabel('Word')
plt.xticks(rotation=30)
plt.ylabel('Count')
plt.title('Top 10 Bigrams in Subject Tags After Removing Stop Words and Lemmatizing', fontsize=10)
# plt.xticks(fontsize=6)
plt.show()


common_words = get_top_n_trigram(data['SUBJECTS'], 10)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['SUBJECTS' , 'count'])
df4.groupby('SUBJECTS').sum()['count'].sort_values(ascending=False)
plt.figure(2)
plt.bar(df4['SUBJECTS'], df4['count'])
plt.xlabel('Word')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.title('Top 10 Trigrams in Subject Tags After Removing Stop Words and Lemmatizing', fontsize=10)
# plt.xticks(fontsize=6)
plt.show()

processed_features = data.iloc[:, 3].values.astype(str)
labels = data.iloc[:, 0].values.astype(str)
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
vectorizer = CountVectorizer(ngram_range=(1, 1)).fit(X_train)
x_test = vectorizer.transform(X_test)
print('X test shape: {}'.format(x_test.shape))
