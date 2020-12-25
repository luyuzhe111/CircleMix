import pandas as pd
import os
import sys
from sklearn.metrics import balanced_accuracy_score, f1_score

root_dir = 'exp_results/config_renal'

c5_config = sys.argv[1]
exp = c5_config.split('Efficientb0')[1].split('.')[0]

c5_config = os.path.join(root_dir, 'Efficientb0' + exp)
c3_config = os.path.join(root_dir, 'Efficientb0-C3' + exp)
c2_config = os.path.join(root_dir, 'Efficientb0-C2' + exp)

c5_pred = os.path.join(c5_config, 'epoch_test.csv')
c3_pred = os.path.join(c3_config, 'prediction.csv')
c2_pred = os.path.join(c2_config, 'prediction.csv')

df_comb = pd.DataFrame(columns=['image', 'target', 'c5_prediction', 'c3_prediction', 'c2_prediction'])

df_c5 = pd.read_csv(c5_pred, index_col=0)
df_c3 = pd.read_csv(c3_pred, index_col=0)
df_c2 = pd.read_csv(c2_pred, index_col=0)

df_comb[['image', 'target', 'c5_prediction']] = df_c5.loc[:, ['image', 'target', 'prediction']].copy()
df_comb['c3_prediction'] = df_c3['prediction']
df_comb['c2_prediction'] = df_c2['prediction']

comb_pred = []
for idx, row in df_comb.iterrows():
    if row['c5_prediction'] == 0 or row['c5_prediction'] == 4:
        comb_pred.append(row['c5_prediction'])
    else:
        if row['c3_prediction'] == 1:
            comb_pred.append(row['c3_prediction'])
        else:
            comb_pred.append(row['c2_prediction'])

df_comb['combined_prediction'] = comb_pred
print('plain accuracy:', balanced_accuracy_score(df_comb['target'], df_comb['c5_prediction']))
f1s = f1_score(df_comb['target'], df_comb['c5_prediction'], average=None)
print('plain f1:', sum(f1s)/len(f1s))

print('hierarchical accuracy:', balanced_accuracy_score(df_comb['target'], df_comb['combined_prediction']))
f1s = f1_score(df_comb['target'], df_comb['combined_prediction'], average=None)
print('hierarchical f1:', sum(f1s)/len(f1s))

df_comb.to_csv(os.path.join(c5_config, 'combined_prediction.csv'), index=False)



