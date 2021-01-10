import pandas as pd
import os
import sys
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

# parent_dir = sys.argv[1]
# parent_config = os.path.basename(parent_dir).split('.')[0]
#
# model_name = parent_config.split('_')[0]
# child_config = model_name + '-C3' + parent_config.split(model_name)[1]
#
# parent_pred = os.path.join(parent_dir, 'epoch_test.csv')
# child_pred = os.path.join(os.path.dirname(parent_dir), child_config, 'prediction.csv')
#
# df_par = pd.read_csv(parent_pred)
# df_chd = pd.read_csv(child_pred)
#
# df_par.sort_values('image')
# df_chd.sort_values('image')
#
# df_comb = pd.DataFrame(columns=['image', 'c5_prediction', 'c3_prediction', 'combined_prediction', 'target'])
# # df_comb = pd.DataFrame(columns=['image', 'c5_prediction', 'c2_prediction', 'combined_prediction', 'target'])
# df_comb[['image', 'c5_prediction']] = df_par.loc[:, ['image', 'prediction']].copy()
# df_comb['target'] = df_par['target'].copy()
# df_comb['c3_prediction'] = df_chd['prediction'].copy()
#
# comb_pred = []
# for idx, row in df_comb.iterrows():
#     if row['c5_prediction'] in [1, 2, 3]:
#         comb_pred.append(row['c3_prediction'])
#     else:
#         comb_pred.append(row['c5_prediction'])
#
# # for idx, row in df_comb.iterrows():
# #     if row['c5_prediction'] in [2, 3]:
# #         comb_pred.append(row['c2_prediction'])
# #     else:
# #         comb_pred.append(row['c5_prediction'])
#
# df_comb['combined_prediction'] = comb_pred
#
# print('plain accuracy:', balanced_accuracy_score(df_comb['target'], df_comb['c5_prediction']))
# f1s = f1_score(df_comb['target'], df_comb['c5_prediction'], average=None)
# print('plain f1:', sum(f1s)/len(f1s))
# print(confusion_matrix(df_comb['target'], df_comb['c5_prediction']))
#
# print('hierarchical accuracy:', balanced_accuracy_score(df_comb['target'], df_comb['combined_prediction']))
# f1s = f1_score(df_comb['target'], df_comb['combined_prediction'], average=None)
# print('hierarchical f1:', sum(f1s)/len(f1s))
# print(confusion_matrix(df_comb['target'], df_comb['combined_prediction']))
#
# output_dir = os.path.dirname(parent_pred)
# output_file = os.path.join(output_dir, 'c3_combined_prediction.csv')
# df_comb.to_csv(output_file, index=False)

parent_dir = sys.argv[1]
parent_config = os.path.basename(parent_dir).split('.')[0]

model_name = parent_config.split('_')[0]
c3_config = model_name + '-C3' + parent_config.split(model_name)[1]
c2_config = model_name + '-C2' + parent_config.split(model_name)[1]

parent_pred = os.path.join(parent_dir, 'epoch_test.csv')
c3_pred = os.path.join(os.path.dirname(parent_dir), c3_config, 'prediction.csv')
c2_pred = os.path.join(os.path.dirname(parent_dir), c2_config, 'prediction.csv')

df_par = pd.read_csv(parent_pred)
df_c3 = pd.read_csv(c3_pred)
df_c2 = pd.read_csv(c2_pred)

df_par.sort_values('image')
df_c3.sort_values('image')
df_c2.sort_values('image')

df_comb = pd.DataFrame(columns=['image', 'c5_prediction', 'c3_prediction', 'c2_prediction', 'combined_prediction', 'target'])
df_comb[['image', 'c5_prediction']] = df_par.loc[:, ['image', 'prediction']].copy()
df_comb['target'] = df_par['target'].copy()
df_comb['c3_prediction'] = df_c3['prediction'].copy()
df_comb['c2_prediction'] = df_c2['prediction'].copy()

comb_pred = []
for idx, row in df_comb.iterrows():
    if row['c5_prediction'] in [1, 2, 3]:
        if row['c3_prediction'] == 1:
            comb_pred.append(row['c3_prediction'])
        else:
            comb_pred.append(row['c2_prediction'])
    else:
        comb_pred.append(row['c5_prediction'])

df_comb['combined_prediction'] = comb_pred

print('plain accuracy:', balanced_accuracy_score(df_comb['target'], df_comb['c5_prediction']))
f1s = f1_score(df_comb['target'], df_comb['c5_prediction'], average=None)
print('plain f1:', sum(f1s)/len(f1s))
print(confusion_matrix(df_comb['target'], df_comb['c5_prediction']))

print('hierarchical accuracy:', balanced_accuracy_score(df_comb['target'], df_comb['combined_prediction']))
f1s = f1_score(df_comb['target'], df_comb['combined_prediction'], average=None)
print('hierarchical f1:', sum(f1s)/len(f1s))
print(confusion_matrix(df_comb['target'], df_comb['combined_prediction']))

output_dir = os.path.dirname(parent_pred)
output_file = os.path.join(output_dir, 'c3c2_combined_prediction.csv')
df_comb.to_csv(output_file, index=False)
