import json
import pandas as pd
import os
import shutil

def main():
    fold_assignment = 'csv/folds_assignment.csv'
    df = pd.read_csv(fold_assignment)

    np_ratio(df, 1)
    np_ratio(df, 2)
    np_ratio(df, 3)
    np_ratio(df, 4)
    np_ratio(df, 5)

    data_dir = '/Data/luy8/centermix/resized_data/train'
    json_dir = 'json'
    json_each_fold(df, 1, data_dir, json_dir)
    json_each_fold(df, 2, data_dir, json_dir)
    json_each_fold(df, 3, data_dir, json_dir)
    json_each_fold(df, 4, data_dir, json_dir)
    json_each_fold(df, 5, data_dir, json_dir)


def np_ratio(df, fold):
    fold1_neg = len(df[(df['fold'] == fold) & (df['target'] == 0)])
    fold1_pos = len(df[(df['fold'] == fold) & (df['target'] == 1)])
    print('In fold %d, there are %d negative, %d positive, and %d in total' % (fold, fold1_neg,
                                                                               fold1_pos, len(df[df['fold'] == fold])))


def json_each_fold(df, fold, data_dir, json_dir):
    root_dir = os.path.dirname(data_dir)
    fold_dir = os.path.join(root_dir, 'fold'+str(fold))
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)

    df_fold = df[df['fold'] == fold]
    data_list = []
    for idx, row in df_fold.iterrows():
        img = row['image_name']
        img_dir = os.path.join(data_dir, img + '.png')
        img_fold_dir = os.path.join(fold_dir, img + '.png')
        shutil.copy(img_dir, img_fold_dir)
        target = row['target']
        patient = row['patient_id']

        one_entry = {'name': img, 'patient': patient, 'image_dir': img_fold_dir, 'target': target}
        data_list.append(one_entry)

    output_json = os.path.join(json_dir, 'fold'+str(fold)+'.json')
    with open(output_json, 'w') as out_file:
        json.dump(data_list, out_file)


if __name__ == '__main__':
    main()
