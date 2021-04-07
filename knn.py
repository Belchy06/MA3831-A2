import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
    dataset = pd.read_csv("lemmed_dat.csv")
    bag_of_words = dataset['SUBJECTS'].values.tolist()
    print(bag_of_words)
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(bag_of_words)

    knn = NearestNeighbors(n_neighbors=2, metric='cosine')
    knn.fit(tfidf)

    input_texts = ['government']
    input_features = vect.transform(input_texts)

    D, N = knn.kneighbors(input_features, n_neighbors=2, return_distance=2)
    for input_text, distances, neighbours in zip(input_texts, D, N):
        print("Input text = ", input_text[:200], "\n")
        for dist, neighbor_idx in zip(distances, neighbours):
            print("Distance = ", dist, "Neighbor idx = ", neighbor_idx)
            print(dataset.iloc[neighbor_idx][:200])
            print("-" * 200)
        print("=" * 200)
        print()
