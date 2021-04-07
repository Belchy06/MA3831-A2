import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

dataset = pd.read_csv('cleaned_dat.csv')
subset2 = dataset.groupby(["COURSENAME"]).count().sort_values(["COMBINED_TITLE"], ascending=False).reset_index()[
        ['COMBINED_TITLE', 'COURSENAME']]
print(subset2)
dat = list(subset2['COMBINED_TITLE'])
print('Floored mean: {}'.format(np.floor(np.mean(dat))))