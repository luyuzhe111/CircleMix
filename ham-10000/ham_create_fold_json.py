from create_fold_json import json_each_fold
import pandas as pd

data_dir = '/Data/luy8/centermix/ham-10000/resized_data'
json_dir = '/Data/luy8/centermix/ham-10000/json'

df = pd.read_csv('/Data/luy8/centermix/ham-10000/csv/folds_assignment.csv', index_col=0)
label_dict = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
targets = []
for index, row in df.iterrows():
    series = row[1:8]
    tar_series = series[series == 1]
    label = label_dict[tar_series.index[0]]
    targets.append(label)

df.insert(8, 'target', targets)

keys = ['lesion_id', 'image', 'target']
for i in range(1, 6):
    json_each_fold(df, i, data_dir, json_dir, keys)

df.to_csv('/Data/luy8/centermix/ham-10000/csv/folds_assignment.csv', index=0)