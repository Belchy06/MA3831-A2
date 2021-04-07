import re
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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


def get_valid_res_input():
    while True:
        res_type = input('Enter resource type ({}): '.format(ALLOWED_RESOURCE_TYPES)).lower()
        if res_type not in ALLOWED_RESOURCE_TYPES:
            print('Invalid resource type!')
        else:
            break
    return res_type


def get_subjects(title, resource_type):
    constructed_string = ""
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
                constructed_string = "".join(subjects)
            elif work_response.status_code == 403:
                print("WORK AUTH ERROR")
            else:
                print("WORK OTHER ERROR")
    elif response.status_code == 403:
        print("AUTH ERROR")
    else:
        print("OTHER ERROR")
    return re.sub('[ ]{2,}', ' ', (constructed_string + " " + title).strip())


if __name__ == '__main__':
    while True:
        title = re.sub('[^A-za-z0-9]+', ' ', input('Enter the new title: ')).lower()
        resource_type = get_valid_res_input()
        subjects = get_subjects(title, resource_type)
        print('Title: {}'.format(title))
        print('Subjects: {}'.format(subjects))
        print('\n\n\n')

        data = pd.read_csv("combined_dat.csv")
        vec = TfidfVectorizer()
        features = vec.fit_transform(data['SUBJECTS'])
        knn = NearestNeighbors(n_neighbors=10, metric='cosine')
        knn.fit(features)

        input_features = vec.transform([' '.join(list(set(subjects.split(' '))))])

        D, N = knn.kneighbors(input_features, n_neighbors=5, return_distance=True)

        for input_text, distances, neighbours in zip(title, D, N):
            print('Input Text: ', title, '\n')
            for dist, neighbour_idx in zip(distances, neighbours):
                print('Distance: ', dist)
                print('Neighbour Index: ', neighbour_idx)
                print(data.iloc[neighbour_idx])
                print("-" * 200)
            print("=" * 200)
            print()
