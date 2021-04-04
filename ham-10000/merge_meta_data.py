import pandas as pd

gh = pd.read_csv('/Data/luy8/centermix/ham-10000/csv/ground_truth.csv')
meta = pd.read_csv('/Data/luy8/centermix/ham-10000/csv/meta_data.csv')

meta_id = meta.iloc[:, 0:2]
df_merged = meta_id.merge(gh, left_on='image', right_on='image')

df_merged.to_csv('/Data/luy8/centermix/ham-10000/csv/ham_data.csv', index=0)