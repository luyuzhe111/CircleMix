from preprocessing import split_folds
import pandas as pd

df = pd.read_csv('csv/ham_data.csv', index_col=0)
patient_col = 'lesion_id'

num_folds = 5
df = split_folds(df, patient_col, num_folds)
df.to_csv('csv/folds_assignment.csv')

keys_to_remove = ['image', 'lesion_id', 'fold']
for i in range(1, num_folds + 1):
    dict_i = dict(df[df['fold'] == i].sum(axis=0, skipna=True).items())
    [dict_i.pop(key, None) for key in keys_to_remove]
    print('fold{} dist:{} total:{}'.format(i, dict_i, sum(list(dict_i.values()))))