from preprocessing import create_fold_file, create_train_file
import random
import pandas as pd


def assign_fold(df, patient_col, num_folds):
    patients = df[patient_col].unique()
    random.seed(0)
    fold_assignment = [random.randint(1, num_folds) for _ in patients]
    fold_dict = {}
    for i in range(len(patients)):
        fold_dict[patients[i]] = fold_assignment[i]

    df['fold'] = 0
    for index, row in df.iterrows():
        df.loc[index, 'fold'] = fold_dict[row[patient_col]]

    return df


def split_fold():
    df = pd.read_csv('csv/ham_data.csv', index_col=0)
    patient_col = 'lesion_id'

    num_folds = 5
    df = assign_fold(df, patient_col, num_folds)
    df.to_csv('csv/folds_assignment.csv')

    keys_to_remove = ['image', 'lesion_id', 'fold']
    for i in range(1, num_folds + 1):
        dict_i = dict(df[df['fold'] == i].sum(axis=0, skipna=True).items())
        [dict_i.pop(key, None) for key in keys_to_remove]
        print('fold{} dist:{} total:{}'.format(i, dict_i, sum(list(dict_i.values()))))

    # create json file for fold
    data_dir = '/skin/resized_data_tmp'
    json_dir = 'json'

    df = pd.read_csv('csv/folds_assignment.csv', index_col=0)
    label_dict = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
    targets = []
    for index, row in df.iterrows():
        series = row[1:8]
        tar_series = series[series == 1]
        label = label_dict[tar_series.index[0]]
        targets.append(label)

    df.insert(8, 'target', targets)

    keys = ['lesion_id', 'image', 'target']
    for i in range(1, 6):
        create_fold_file(df, i, data_dir, json_dir, keys)

    df.to_csv('csv/folds_assignment.csv', index=0)


def create_trainset():
    output_dir = 'json'
    for i in range(1, 6):
        fold_i = f'{output_dir}/fold{i}.json'
        j = i + 1
        k = i + 2

        if j > 5:
            j = j % 5

        if k > 5:
            k = k % 5

        fold_j = f'{output_dir}/fold{j}.json'
        fold_k = f'{output_dir}/fold{k}.json'

        print(fold_i, fold_j, fold_k)

        create_train_file(fold_i, fold_j, fold_k, f'trainset{i}', output_dir)


if __name__ == "__main__":
    split_fold()
    create_trainset()