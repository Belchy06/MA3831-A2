import re
import time
import nltk
import requests
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from lemmatizer import Lemmatizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

ALLOWED_RESOURCE_TYPES = [
    "book",
    "journal",
    "article"
]

API_KEYS = [
    # 'movpr4q69afm09i8',
    'ah6bttobqufrrmma',
    # '1vrmlm0pm85uftsk',
    # 'ei53b8ij7m6vt7v7'
]

SEARCH_URL = 'https://api.trove.nla.gov.au/v2/result'
WORK_URL = 'https://api.trove.nla.gov.au/v2'
NUM_RESULTS = 5


def get_valid_res_input():
    while True:
        res_type = input('Enter resource type ({}): '.format(ALLOWED_RESOURCE_TYPES)).lower()
        if res_type not in ALLOWED_RESOURCE_TYPES:
            print('Invalid resource type!')
        else:
            break
    return res_type


def get_resource(title, resource_type):
    constructed_string = ""
    result_title = ""
    api_key = API_KEYS[0]
    params = {
        'q': '"' + title + "'",
        'key': api_key,
        'n': 1,
        'encoding': 'json',
        'zone': resource_type
    }
    response = requests.get(SEARCH_URL, params=params)
    if response.status_code == requests.codes.ok:
        response_data = response.json()
        # print(json.dumps(response_data, indent=2))
        zone = response_data['response']['zone'][0]
        if int(zone['records']['n']) > 0:
            result_title = re.sub('[^A-za-z0-9]+', ' ', zone['records']['work'][0]['title']).lower()
            work_id = zone['records']['work'][0]['url']
            params = {
                'key': api_key,
                'reclevel': 'full',
                'encoding': 'json'
            }
            work_response = requests.get(WORK_URL + work_id, params=params)
            if work_response.status_code == requests.codes.ok:
                work_data = work_response.json()
                subjects = []
                try:
                    # If the subjects field is a list of subject, append each subject to the list of subjects
                    if isinstance(work_data['work']['subject'], list):
                        for subject in work_data['work']['subject']:
                            subject = re.sub('[^A-za-z0-9]+', ' ', subject).lower()
                            subjects.append(subject)
                    # Else the subject field is a string so just append the entire string
                    else:
                        subject = re.sub('[^A-za-z0-9]+', ' ', work_data['work']['subject']).lower()
                        subjects.append(subject)
                except KeyError:
                    pass
                constructed_string = " ".join(subjects)
            elif work_response.status_code == 403:
                print("WORK AUTH ERROR")
            else:
                print("WORK OTHER ERROR")
    elif response.status_code == 403:
        print("AUTH ERROR")
    else:
        print("OTHER ERROR")
    return [re.sub('[ ]{2,}', ' ', (constructed_string + " " + title).strip()), result_title]


if __name__ == '__main__':
    lem = Lemmatizer()
    while True:
        title = re.sub('[^A-za-z0-9]+', ' ', input('Enter the new title: ')).lower()
        resource_type = get_valid_res_input()
        print('Getting resource info...')
        subjects, result_title = get_resource(title, resource_type)
        subjects = ' '.join([subject for subject in subjects.split(' ') if subject not in stopwords.words('english')])
        subjects = lem.lemmatize_sentence(subjects)
        subjects = ' '.join(list(set(word_tokenize(subjects))))
        print('Resource found:')
        print('Title: {}'.format(result_title))
        print('Subjects: {}'.format(subjects))
        print('\n')

        data = pd.read_csv("lemmed_dat.csv")
        vec = TfidfVectorizer()
        features = vec.fit_transform(data['SUBJECTS'])
        knn = NearestNeighbors(n_neighbors=10, metric='cosine')
        knn.fit(features)
        input_features = vec.transform([' '.join(list(set(subjects.split(' '))))])

        start_time = time.time()
        D, N = knn.kneighbors(input_features, n_neighbors=NUM_RESULTS, return_distance=True)
        print('Found {} solutions in {} seconds'.format(NUM_RESULTS, time.time() - start_time))
        for input_text, distances, neighbours in zip(title, D, N):
            print('Input Text: ', subjects, '\n')
            for dist, neighbour_idx in zip(distances, neighbours):
                print("-" * 200)
                print('Distance: ', dist)
                print('Neighbour Index: ', neighbour_idx)
                print(data.iloc[neighbour_idx])
            print("=" * 200)
            print()
