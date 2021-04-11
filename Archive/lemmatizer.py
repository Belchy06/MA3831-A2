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


class Lemmatizer:
    def __init__(self):
        print(wn.__class__)  # <class 'nltk.corpus.util.LazyCorpusLoader'>
        wn.ensure_loaded()  # first access to wn transforms it
        print(wn.__class__)

    def lemmatize_sentence(self, sentence):
        tagged_sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
        tagged_wordnet = map(lambda x: (x[0], self.get_wordnet_pos(x[1])), tagged_sentence)
        lemmatized_sentence = []
        for word, tag in tagged_wordnet:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(list(set(lemmatized_sentence)))

    def get_wordnet_pos(self, tag):
        """Map POS tag to first character lemmatize() accepts"""
        tag_dict = {"J": wn.ADJ,
                    "N": wn.NOUN,
                    "V": wn.VERB,
                    "R": wn.ADV}
        return tag_dict.get(tag[0], wn.NOUN)
