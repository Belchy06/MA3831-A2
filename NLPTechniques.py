import csv
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('periodicals')

# 1. Stopword removal
# 2. Lemmatization
# 3. N-grmas
# 4. Vectorization

# Finding keywords that span the corpus
# 5. Principal Component Analysis / Latent Semantic Analysis -> Word themes
# or
# 5. Decision Tree


def remove_stopwords(dataset):
    for index, resource in dataset.iterrows():
        dataset.loc[index, 'SUBJECTS'] = ' '.join([subject for subject in resource.SUBJECTS.split(' ') if subject not in stopwords])
    return dataset


def lemmatize(dataset):
    for index, resource in dataset.iterrows():
        tagged_sentence = nltk.pos_tag(nltk.word_tokenize(resource.SUBJECTS))
        tagged_wordnet = map(lambda x: (x[0], get_wordnet_pos(x[1])), tagged_sentence)
        lemmatized_sentence = []
        for word, tag in tagged_wordnet:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        dataset.loc[index, 'SUBJECTS'] = " ".join(list(set(lemmatized_sentence)))
    return dataset


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}
    return tag_dict.get(tag[0], wn.NOUN)


if __name__ == '__main__':
    data = pd.read_csv('unique_resources.csv')
    data = remove_stopwords(data)
    data = lemmatize(data)
    print(data)
    data.to_csv("unique_NLP_resources.csv", index=False, encoding='utf-8-sig')



