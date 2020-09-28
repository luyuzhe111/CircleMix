import json
import pandas as pd


def main():
    fold_assignment = 'csv/folds_assignment.csv'
    df = pd.read_csv(fold_assignment)

    np_ratio(df, 1)
    np_ratio(df, 2)
    np_ratio(df, 3)
    np_ratio(df, 4)
    np_ratio(df, 5)


def np_ratio(df, fold):
    fold1_neg = len(df[(df['fold']==fold) & (df['target']==0)])
    fold1_pos = len(df[(df['fold']==fold) & (df['target']==1)])
    print('In fold %d, there are %d negative, %d positive, and %d in total' % (fold, fold1_neg,
                                                                               fold1_pos, len(df[df['fold']==fold])))


if __name__ == '__main__':
    main()