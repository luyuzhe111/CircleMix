from preprocessing import *
import pandas as pd

data_dir = 'centermix/ham-10000/resized_data'
json_dir = 'centermix/ham-10000/json'

df = pd.read_csv('ham-10000/csv/folds_assignment.csv', index_col=0)
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
    create_fold_file(df, i, data_dir, json_dir, keys)

df.to_csv('csv/folds_assignment.csv', index=0)