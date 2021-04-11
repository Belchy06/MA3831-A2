import nltk
import pandas as pd
from nltk.corpus import stopwords
from lemmatizer import Lemmatizer

nltk.download('stopwords')

if __name__ == '__main__':
    lem = Lemmatizer()
    raw_data = pd.read_csv('cleaned_dat.csv')
    # raw_data = raw_data.iloc[0:6,:]
    raw_data['SUBJECTS'] = [lem.lemmatize_sentence(
        ' '.join([subject for subject in subjects.split(' ') if subject not in stopwords.words('english')])) for subjects in
                        raw_data['SUBJECTS'].tolist()]
    z = [(pd.Series(nltk.ngrams(x, 2)).value_counts())[:10] for x in [y.split(' ') for y in [sentence for sentence in raw_data['SUBJECTS']]]]
    # raw_data['SUBJECTS'] = [subject for subject in  if subject not in stopwords.words('english')]
    print()
