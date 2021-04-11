import re
import os
import csv
import time
import requests
import threading
import traceback
import pandas as pd


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
                    constructed_string = " ".join(subjects)
                elif work_response.status_code == 403:
                    print("Thread {}, Index {}: WORK AUTH ERROR".format(index, i))
                else:
                    print("Thread {}, Index {}: WORK OTHER ERROR".format(index, i))
            # Remove punctuation
            # constructed_string = constructed_string + " " + title
            constructed_string = re.sub('[ ]{2,}', ' ', constructed_string.strip())
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
        # Remove any non letter or number character except ' and " and replace with ' '
        data[column] = [re.sub("[^a-z0-9\'\"]", ' ', elem.lower()) for elem in data[column].tolist()]
        # Remove any ' and "
        data[column] = [re.sub("[\'\"]", '', elem) for elem in data[column].tolist()]
        # Remove any \ufffd char (Unicode Character 'REPLACEMENT CHARACTER')
        data[column] = [elem.replace(u'\ufffd', ' ') for elem in data[column].tolist()]
        # Replace any existence of 2 or more ' ' with a single ' '
        data[column] = [re.sub('[ ]{2,}', ' ', elem) for elem in data[column].tolist()]
        # Remove any leading or trailing ' '
        data[column] = [elem.strip() for elem in data[column].tolist()]
    data = data[data['RESOURCE_TYPE'].isin(ALLOWED_RESOURCE_TYPES)]
    return data


def remove_duplicates(data):
    dat = data.sort_values(by='COURSENAME').copy()
    dat.drop_duplicates(keep='first', inplace=True)
    return dat


def merge_datasets(courses, resources):
    resource_data = resources[['COMBINED_TITLE', 'SUBJECTS']]
    course_data = courses[['COURSENAME', 'COMBINED_TITLE', 'RESOURCE_TYPE']]
    for index, resource in course_data.iterrows():
        title = resource.COMBINED_TITLE
        resource_subjects = resource_data[resource_data['COMBINED_TITLE'] == title]['SUBJECTS']
        course_data.loc[index, 'SUBJECTS'] = resource_subjects.values[0]
    return course_data


def get_subset_based_on_title(data):
    # resource has more than 1 associated course
    subset1 = data.groupby(["COMBINED_TITLE"]).count().sort_values(["COURSENAME"], ascending=False).reset_index()[
        ['COMBINED_TITLE', 'COURSENAME']]
    ss1 = list(set(subset1[subset1['COURSENAME'] >= 2]['COMBINED_TITLE']))
    subset_based_on_title = data[data['COMBINED_TITLE'].isin(ss1)]
    return subset_based_on_title


def get_subset_based_on_course(data):
    # Course has more than 4 associated resources
    subset2 = data.groupby(["COURSENAME"]).count().sort_values(["COMBINED_TITLE"], ascending=False).reset_index()[
        ['COURSENAME', 'COMBINED_TITLE']]
    ss2 = list(set(subset2[subset2['COMBINED_TITLE'] >= 5]['COURSENAME']))
    subset_based_on_course = data[data['COURSENAME'].isin(ss2)]
    return subset_based_on_course


if __name__ == "__main__":
    """
    DATA PRE-PROCESSING
    """
    if not os.path.isfile('resource_data.csv'):
        dataset = pd.read_csv("raw_data.csv")
        if not os.path.isfile('cleaned_data.csv'):
            preprocessed_data = preprocess_data(dataset)
            cleaned_data = remove_duplicates(preprocessed_data)
            cleaned_data.to_csv("cleaned_data.csv", index=False, encoding='utf-8-sig')
            # Select only the rows that contain a unique COMBINED_TITLE
            unique_resources = cleaned_data.drop_duplicates(subset=["COMBINED_TITLE"])
            print('Original data length: {}'.format(len(dataset)))
            print('Preprocessed data length: {}'.format(len(preprocessed_data)))
            print('Cleaned data length: {}'.format(len(cleaned_data)))
            print('Unique resources: {}'.format(len(unique_resources)))



        """
        MULTI-THREADING
        """
        unique_resources = pd.read_csv('cleaned_data.csv').drop_duplicates(subset=['COMBINED_TITLE'])
        # Split dataframe into equal amounts
        n = int(len(unique_resources) / MAX_THREADS) + 1
        list_df = [unique_resources[i:i + n] for i in range(0, unique_resources.shape[0], n)]
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
        combined_resources = pd.concat([pd.read_csv(f) for f in all_files])
        # export to csv
        combined_resources.to_csv("resource_data.csv", index=False, encoding='utf-8-sig')
        # Remove temp files
        for file in all_files:
            os.remove(file)

    """
    MERGE RESOURCE AND COURSE FILES
    """
    if not os.path.isfile('merged_data.csv') \
            and os.path.isfile('resource_data.csv') \
            and os.path.isfile('cleaned_data.csv'):

        cleaned_data = pd.read_csv('cleaned_data.csv')
        combined_resources = pd.read_csv('resource_data.csv')

        merged_dataset = merge_datasets(cleaned_data, combined_resources)
        merged_dataset.to_csv("merged_data.csv", index=False, encoding='utf-8-sig')

    """
    SUBSET THE MERGED DATA SET
    """
    if not os.path.isfile('title_subset.csv') and not os.path.isfile('course_subset.csv'):
        dataset = pd.read_csv('merged_data.csv')
        print('Original: {}'.format(len(dataset)))
        title_sub = get_subset_based_on_title(dataset.copy())
        title_sub.to_csv("title_subset.csv", index=False, encoding='utf-8-sig')
        print('Title subset: {}'.format(len(title_sub)))
        course_sub = get_subset_based_on_course(dataset.copy())
        course_sub.to_csv("course_subset.csv", index=False, encoding='utf-8-sig')
        print('Course subset: {}'.format(len(course_sub)))
