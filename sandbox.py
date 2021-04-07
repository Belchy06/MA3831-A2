import csv
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    # print(tag)
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}

    return tag_dict.get(tag[0], wn.NOUN)


def lemmatize_sentence(sentence):
    tagged_sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
    tagged_wordnet = map(lambda x: (x[0], get_wordnet_pos(x[1])), tagged_sentence)
    lemmatized_sentence = []
    for word, tag in tagged_wordnet:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(list(set(lemmatized_sentence)))


if __name__ == "__main__":
    dataset = pd.read_csv("combined_dat.csv")

    # Ensure wordnet corpus has loaded
    print(wn.__class__)  # <class 'nltk.corpus.util.LazyCorpusLoader'>
    wn.ensure_loaded()  # first access to wn transforms it
    print(wn.__class__)

    for index, row in dataset.iterrows():
        words = list(set(word_tokenize(row['SUBJECTS'])))
        words = ' '.join([word for word in words if word not in stopwords.words('english')])
        lemmed_words = lemmatize_sentence(words)
        row['SUBJECTS'] = lemmed_words

    # Combine files
    with open(r'lemmed_dat.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['COURSENAME', 'COMBINED_TITLE', 'RESOURCE_TYPE', 'SUBJECTS'])

    # export to csv
    dataset.to_csv("lemmed_dat.csv", index=False, encoding='utf-8-sig')
