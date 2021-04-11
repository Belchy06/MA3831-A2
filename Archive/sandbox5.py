import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC

if __name__ == '__main__':
    data = pd.read_csv('course_subset.csv')
    processed_features = data.iloc[:, 3].values
    labels = data.iloc[:, 0].values
    # print(processed_features)

    # train, test split, the train data is just for TfidfVectorizer() fit
    x_train, x_test, y_train, y_test = train_test_split(processed_features, labels, train_size=0.75, random_state=0)
    tfidf = TfidfVectorizer()
    tfidf.fit(x_train)

    # vectorizer test data for 5-fold cross-validation
    x_test = tfidf.transform(x_test)

    scoring = ['accuracy']
    clf = SVC(kernel='linear')  # using support vec machine
    scores = cross_validate(clf, x_test, y_test, scoring=scoring, cv=5, return_train_score=False)
    print(scores)
