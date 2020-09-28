import pandas as pd
import random


def main():
    df = pd.read_csv('train.csv')
    patients = df['patient_id'].unique()
    fold_assignment = [random.randint(1, 5) for i in patients]
    fold_dict = {}
    for i in range(len(patients)):
        fold_dict[patients[i]] = fold_assignment[i]

    df['fold'] = 0
    for index, row in df.iterrows():
        df.loc[index, 'fold'] = fold_dict[row['patient_id']]

    print('Number of images in fold 1', len(df[df['fold'] == 1]))
    print('Number of images in fold 2', len(df[df['fold'] == 2]))
    print('Number of images in fold 3', len(df[df['fold'] == 3]))
    print('Number of images in fold 4', len(df[df['fold'] == 4]))
    print('Number of images in fold 5', len(df[df['fold'] == 5]))

    print(df.head())

    df.to_csv('folds_assignment.csv')


if __name__ == '__main__':
    main()

