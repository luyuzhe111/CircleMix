import json
import pandas as pd
import os
import shutil


def main():
    fold_assignment = 'csv/folds_assignment.csv'
    df = pd.read_csv(fold_assignment, index_col=0)

    for i in range(1, 6):
        np_ratio(df, i)

    data_dir = '/Data/luy8/centermix/resized_data'
    json_dir = 'json'
    keys = ['patient_id', 'image_name', 'target']
    for i in range(1, 6):
        json_each_fold(df, i, data_dir, json_dir, keys)


def np_ratio(df, fold):
    fold1_neg = len(df[(df['fold'] == fold) & (df['target'] == 0)])
    fold1_pos = len(df[(df['fold'] == fold) & (df['target'] == 1)])
    print('In fold %d, there are %d negative, %d positive, and %d in total' % (fold, fold1_neg,
                                                                               fold1_pos, len(df[df['fold'] == fold])))


def json_each_fold(df, fold, data_dir, json_dir, keys):
    patient_id, image_name, label = keys
    fold_dir = os.path.join(data_dir, 'fold'+str(fold))
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)

    df_fold = df[df['fold'] == fold]
    data_list = []
    for index, row in df_fold.iterrows():
        img = index
        img_dir = os.path.join(data_dir, img + '.png')
        img_fold_dir = os.path.join(fold_dir, img + '.png')
        # shutil.copy(img_dir, img_fold_dir)
        target = row[label]
        patient = row[patient_id]

        one_entry = {'name': img, 'patient': patient, 'image_dir': img_fold_dir, 'target': target}
        data_list.append(one_entry)

    output_json = os.path.join(json_dir, 'fold'+str(fold)+'.json')
    with open(output_json, 'w') as out_file:
        json.dump(data_list, out_file)


if __name__ == '__main__':
    main()
