import pandas as pd
import re

ALLOWED_RESOURCE_TYPES = [
    "book",
    "journal",
    "article"
]

data = pd.read_csv("data.csv")
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
data.to_csv('testing.csv', index=False, encoding='utf-8-sig')
