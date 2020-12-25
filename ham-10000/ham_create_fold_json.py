from create_fold_json import json_each_fold
import pandas as pd

data_dir = '/Data/luy8/centermix/ham-10000/resized_data'
json_dir = '/Data/luy8/centermix/ham-10000/json'

df = pd.read_csv()
label_dict = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
for index, row in df.iterrows():
    print(row)
    print()
#
# keys = ['lesion_id', 'image', 'target']
#
# for i in range(1, 6):
#     json_each_fold(df, i, data_dir, json_dir, keys)