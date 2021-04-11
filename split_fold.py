import random

'''
This function split samples into specified number of folds at subject level
patient_col is a list of patient ids for each sample
'''

def split_folds(df, patient_col, num_folds):
    patients = df[patient_col].unique()
    fold_assignment = [random.randint(1, num_folds) for _ in patients]
    fold_dict = {}
    for i in range(len(patients)):
        fold_dict[patients[i]] = fold_assignment[i]

    df['fold'] = 0
    for index, row in df.iterrows():
        df.loc[index, 'fold'] = fold_dict[row[patient_col]]

    return df

