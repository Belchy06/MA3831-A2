import csv
import nltk
import pandas as pd
import re
from lemmatizer import Lemmatizer
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


if __name__ == "__main__":
    print(sw.__class__)  # <class 'nltk.corpus.util.LazyCorpusLoader'>
    sw.ensure_loaded()  # first access to wn transforms it
    print(sw.__class__)

    lem = Lemmatizer()
    data = pd.read_csv('test.csv')
    data['SUBJECTS'] = [lem.lemmatize_sentence(' '.join([subject for subject in subjects.split(' ') if subject not in sw.words('english')])) for subjects in data['SUBJECTS'].tolist()]
    with open(r'lemmed_dat2.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['COURSENAME', 'COMBINED_TITLE', 'RESOURCE_TYPE', 'SUBJECTS'])
    # export to csv
    data.to_csv("lemmed_dat2.csv", index=False, encoding='utf-8-sig')



    # data = pd.read_csv('preprocessed_data.csv')
    # for column in data:
    #     # data[column] = [re.sub(r'[^A-za-z0-9]+', ' ', elem) for elem in data[column].tolist()]
    #     # data[column] = [re.sub(r'[\[\]]', ' ', elem) for elem in data[column].tolist()]
    #     # data[column] = [re.sub('[ ]{2,}', ' ', elem) for elem in data[column].tolist()]
    #     data[column] = [elem.strip() for elem in data[column].tolist()]
    # data.to_csv("cleaned_preprocessed_dat.csv", index=False, encoding='utf-8-sig')
