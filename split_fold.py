import pandas as pd
import random


def split_folds(df, patient_col):
    patients = df[patient_col].unique()
    fold_assignment = [random.randint(1, 5) for i in patients]
    fold_dict = {}
    for i in range(len(patients)):
        fold_dict[patients[i]] = fold_assignment[i]

    df['fold'] = 0
    for index, row in df.iterrows():
        df.loc[index, 'fold'] = fold_dict[row[patient_col]]

    return df


if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    patient_col = 'patient_id'

    df = split_folds(df, patient_col)
    for i in range(1, 6):
        print('Number of images in fold {}'.format(i), len(df[df['fold'] == i]))

    print(df.head())
    df.to_csv('/csv/folds_assignment.csv')

