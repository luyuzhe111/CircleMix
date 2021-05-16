import pandas as pd

gh = pd.read_csv('skin/csv/ground_truth.csv')
meta = pd.read_csv('skin/csv/meta_data.csv')

meta_id = meta.iloc[:, 0:2]
df_merged = meta_id.merge(gh, left_on='image', right_on='image')

df_merged.to_csv('skin/csv/ham_data.csv', index=0)