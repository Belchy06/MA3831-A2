import csv
import pandas as pd
import re

dataset = pd.read_csv('combined_dat.csv')
for column in dataset:
    dataset[column] = [re.sub('[ ]{2,}', ' ', elem) for elem in dataset[column].tolist()]
with open(r'cleaned_dat.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['COURSENAME', 'COMBINED_TITLE', 'RESOURCE_TYPE', 'SUBJECTS'])
dataset.to_csv('cleaned_dat.csv', index=False, encoding='utf-8-sig')
