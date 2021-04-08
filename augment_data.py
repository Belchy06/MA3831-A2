import csv
import nltk
import pandas as pd
import threading
import time
import re
import requests
import string
import os
import traceback

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


PRINTABLE = set(string.printable)
API_KEYS = [
    'movpr4q69afm09i8',
    'ah6bttobqufrrmma',
    '1vrmlm0pm85uftsk',
    'ei53b8ij7m6vt7v7'
]
ALLOWED_RESOURCE_TYPES = [
    "book",
    "journal",
    "article"
]
MAX_THREADS = 8
SEARCH_URL = "https://api.trove.nla.gov.au/v2/result"
WORK_URL = "https://api.trove.nla.gov.au/v2"


def thread_function(index, data):
    api_key = API_KEYS[index % len(API_KEYS)]
    with open(r'output{}.csv'.format(index), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['COURSENAME', 'COMBINED_TITLE', 'RESOURCE_TYPE', 'SUBJECTS'])

    length = len(data)
    i = 0
    for row in data.iterrows():
        print("Thread {}: Index {}/{}".format(index, i, length))
        i = i + 1
        title = row[1].COMBINED_TITLE
        res_type = row[1].RESOURCE_TYPE
        params = {
            'q': '"' + title + "'",
            'key': api_key,
            'n': 1,
            'encoding': 'json'
        }
        if res_type == "book":
            params['zone'] = 'book'
        else:
            params['zone'] = "article"

        response = requests.get(SEARCH_URL, params=params)
        if response.status_code == requests.codes.ok:
            constructed_string = ""
            response_data = response.json()
            # formatted_data = json.dumps(response_data, indent=2)
            zone = response_data['response']['zone'][0]
            if int(zone['records']['n']) > 0:
                # result_title = re.sub('[^A-za-z0-9]+', ' ', zone['records']['work'][0]['title']).lower()
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
                    print("Thread {}, Index {}: WORK AUTH ERROR".format(index, i))
                else:
                    print("Thread {}, Index {}: WORK OTHER ERROR".format(index, i))
            # Remove punctuation
            constructed_string = constructed_string + " " + re.sub('[^A-za-z0-9]+', ' ', title).lower()

            fields = [row[1].COURSENAME,  # COURSENAME
                      title,  # COMBINED_TITLE
                      res_type,  # RESOURCE_TYPE
                      constructed_string]  # SUBJECTS
            with open(r'output{}.csv'.format(index), 'a', newline='') as file:
                w = csv.writer(file)
                try:
                    w.writerow(fields)
                except Exception:
                    print("Thread {}, Index {}: MISSED WRITING TO FILE. Title: {}".format(index, i, title))
                    traceback.print_exc()
        elif response.status_code == 403:
            print("Thread {}, Index {}: AUTH ERROR".format(index, i))
        else:
            print("Thread {}, Index {}: OTHER ERROR".format(index, i))
        time.sleep(0.8)


def preprocess_data(data):
    data = data[["COURSENAME", "TITLE", "RESOURCE_TYPE", "SUBTITLE"]]
    data['COMBINED_TITLE'] = data.apply(
        lambda x: '%s_%s' % (x['TITLE'], x['SUBTITLE']) if (x['TITLE'] != x['SUBTITLE']) else x['TITLE'], axis=1)
    data = data[['COURSENAME', 'COMBINED_TITLE', 'RESOURCE_TYPE']]
    data = data[-data['RESOURCE_TYPE'].isnull()]
    for column in data:
        data[column] = [re.sub("[\"']", '', elem.lower()) for elem in data[column].tolist()]
        data[column] = [elem.replace("\n", " ") for elem in data[column].tolist()]
        data[column] = [re.sub(r'[^A-za-z0-9]+', ' ', elem) for elem in data[column].tolist()]
        data[column] = [re.sub(r'[\[\]]', ' ', elem) for elem in data[column].tolist()]
        data[column] = [elem.replace(u'\ufffd', '') for elem in data[column].tolist()]
        data[column] = [re.sub('[ ]{2,}', ' ', elem) for elem in data[column].tolist()]
    data = data[data['RESOURCE_TYPE'].isin(ALLOWED_RESOURCE_TYPES)]
    return data


def subset_data(data):
    # Basically resource that appears in more than two courses
    subset1 = data.groupby(["COMBINED_TITLE"]).count().sort_values(["COURSENAME"], ascending=False).reset_index()[
        ['COMBINED_TITLE', 'COURSENAME']]
    ss1 = list(set(subset1[subset1['COURSENAME'] >= 2]['COMBINED_TITLE']))
    subset_based_on_title = data[data['COMBINED_TITLE'].isin(ss1)]
    print("Original # of titles: " + str(len(data)))
    print("Subset # of titles: " + str(len(subset_based_on_title)))
    print("Subset unique # of titles: " + str(len(subset_based_on_title['COMBINED_TITLE'].unique())))
    print("Subset unique # of courses: " + str(len(subset_based_on_title['COURSENAME'].unique())))
    return subset_based_on_title


def strip_whitespace(data):
    for column in data:
        data[column] = [elem.strip() for elem in data[column].tolist()]
        data[column] = [re.sub('[ ]{2,}', ' ', elem) for elem in data[column].tolist()]
    return data


if __name__ == "__main__":
    dataset = pd.read_csv("data.csv")
    processed_data = preprocess_data(dataset)
    processed_data.to_csv("preprocessed_data.csv", index=False, encoding='utf-8-sig')
    print('Original length: {}'.format(len(processed_data)))
    # subset_based_on_title = subset_data(dataset)
    # Select only the rows that contain a unique COMBINED_TITLE
    unique_data = processed_data.drop_duplicates(subset=["COMBINED_TITLE"])
    print('Unique resources: {}'.format(len(unique_data)))

    # thread_function(1, unique_data.iloc[0:2])
    """
    MULTI-THREADING
    """
    # Split dataframe into equal amounts
    n = int(len(unique_data) / MAX_THREADS) + 1
    list_df = [unique_data[i:i + n] for i in range(0, unique_data.shape[0], n)]
    # Instantiate threads
    threads = list()
    for idx in range(MAX_THREADS):
        print("Main: creating and starting thread {}".format(idx))
        x = threading.Thread(target=thread_function, args=(idx, list_df[idx],))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()
        print("Main: \tThread {} done".format(index))

    """
    COMBINING TEMP FILES
    """
    # Combine files
    with open(r'resource_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['COURSENAME', 'COMBINED_TITLE', 'RESOURCE_TYPE', 'SUBJECTS'])

    all_files = ['output{}.csv'.format(i) for i in range(MAX_THREADS)]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_files])
    combined_csv = strip_whitespace(combined_csv)
    # export to csv
    combined_csv.to_csv("resource_data.csv", index=False, encoding='utf-8-sig')
    # Remove temp files
    for file in all_files:
        os.remove(file)

    """
    MERGE RESOURCE AND COURSE FILES
    """
    resource_data = combined_csv
    course_data = processed_data
    resource_data = resource_data[['COMBINED_TITLE', 'SUBJECTS']]
    course_data = course_data[['COURSENAME', 'COMBINED_TITLE', 'RESOURCE_TYPE']]
    for index, resource in course_data.iterrows():
        title = resource[1]
        resource_subjects = resource_data[resource_data['COMBINED_TITLE'] == title]['SUBJECTS']
        course_data.loc[index, 'SUBJECTS'] = resource_subjects.values[0]
    course_data.to_csv("merged_data.csv", index=False, encoding='utf-8-sig')



    """
    AGGREGATE COURSE SUBJECTS
    """
    # dataset = pd.read_csv("combined_dat.csv")
    # dataset = dataset[['COURSENAME', 'SUBJECTS']]
    # new_ds = dataset.iloc[:]
    # new_df = new_ds.groupby(new_ds['COURSENAME'], as_index=False).agg(' '.join)
    # new_df.to_csv("course_subset.csv", index=False, encoding='utf-8-sig')
