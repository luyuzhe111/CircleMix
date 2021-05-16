from PIL import Image
import os
import json
import pandas as pd
import random
import shutil
import numpy as np


'''
this function performs center crop as pre-processing
'''
def crop_center(input_dir, output_dir, new_size):
    for idx, image in enumerate(os.listdir(input_dir)):
        print(idx)
        img = Image.open(os.path.join(input_dir, image))
        width, height = img.size
        crop_size = min(width, height)

        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2

        cropped_img = img.crop((left, top, right, bottom))
        resized_img = cropped_img.resize((new_size, new_size))

        output_img_dir = os.path.join(output_dir, image.split('.')[0] + '.png')
        resized_img.save(output_img_dir)


'''
create json file from csv file
'''
def create_data_file(train_csv, data_dir, output_dir):
    df = pd.read_csv(train_csv)

    data_list = []
    for index, row in df.iterrows():
        img = row['image_name']
        img_dir = os.path.join(data_dir, img + '.png')
        target = row['target']

        one_entry = {'name': img, 'image_dir': img_dir, 'target': target}
        data_list.append(one_entry)

    output_json = os.path.join(output_dir, 'train.json')
    with open(output_json, 'w') as out_file:
        json.dump(data_list, out_file)


'''
check whether there is contamination between folds
'''
def check_contam(fold_dir, dataset):
    fold_lst = []
    num_folds = 5
    for i in range(num_folds):
        with open(f'{fold_dir}/fold{i + 1}.json') as f:
            data = json.load(f)
            patient = [i['patient'] for i in data]
            fold_lst.append(patient)


    print(f'Checking {dataset} dataset...')
    for i in range(num_folds):
        for j in range(i + 1, num_folds):
            cur_fold = set(fold_lst[i])
            next_fold = set(fold_lst[j])
            contam = cur_fold.intersection(next_fold)
            print(f'Between {i+1} and {j+1} contam:', contam)


'''
create a json file for each fold
'''
def create_fold_file(df, fold, data_dir, json_dir, keys):
    patient_id, image_name, label = keys

    df_fold = df[df['fold'] == fold]
    data_list = []
    for index, row in df_fold.iterrows():
        img = index
        img_dir = os.path.join(data_dir, img + '.png')
        target = row[label]
        patient = row[patient_id]

        one_entry = {'subj': patient, 'image': img, 'image_dir': img_dir, 'target': target}
        data_list.append(one_entry)

    output_json = os.path.join(json_dir, 'fold'+str(fold)+'.json')
    with open(output_json, 'w') as out_file:
        json.dump(data_list, out_file)


'''
create training set from a subset of folds
'''
def create_train_file(fold_1, fold_2, fold_3, train_set, output_dir):
    data_set = []
    with open(fold_1, 'r') as json_file:
        data1 = json.load(json_file)
        data_set += data1

    with open(fold_2, 'r') as json_file:
        data2 = json.load(json_file)
        data_set += data2

    with open(fold_3, 'r') as json_file:
        data3 = json.load(json_file)
        data_set += data3

    fname = train_set + '.json'
    with open(os.path.join(output_dir, fname), 'w') as output_file:
        json.dump(data_set, output_file)


'''
The following four functions perform re-sampling techniques before training
'''
def undersample_binary(input_json, output_dir, fold):
    with open(input_json) as f:
        data = json.load(f)

    neg_data = [item for item in data if item['target'] == 0]
    pos_data = [item for item in data if item['target'] == 1]

    max_len = len(pos_data)
    us_neg_data = list(np.random.choice(neg_data, max_len))
    us_data = us_neg_data + pos_data

    print(len(us_neg_data), len(us_data))

    output_fname = fold + '_us.json'
    output_fdir = os.path.join(output_dir, output_fname)
    with open(output_fdir, 'w') as f:
        json.dump(us_data, f)


def undersample_multi(input_json, output_dir, fold, num_class):
    with open(input_json) as f:
        data = json.load(f)

    us_data = []
    folds_data = []
    for i in range(num_class):
        data_fold = [item for item in data if item['target'] == i]
        folds_data.append(data_fold)

    data_folds_length = [len(fold) for fold in folds_data]
    min_len = min(data_folds_length)

    for fold_data in folds_data:
        us_fold_data = list(np.random.choice(fold_data, min_len))
        us_data += us_fold_data

    print('Downsampling dataset size', len(us_data))
    output_fname = fold + '_us.json'
    output_fdir = os.path.join(output_dir, output_fname)
    with open(output_fdir, 'w') as f:
        json.dump(us_data, f)


def oversample_binary(input_json, output_dir, fold):
    with open(input_json) as f:
        data = json.load(f)

    neg_data = [item for item in data if item['target'] == 0]
    pos_data = [item for item in data if item['target'] == 1]

    max_len = len(neg_data)
    os_pos_data = list(np.random.choice(pos_data, max_len))
    os_data = os_pos_data + neg_data

    output_fname = fold + '_os.json'
    output_fdir = os.path.join(output_dir, output_fname)
    with open(output_fdir, 'w') as f:
        json.dump(os_data, f)


def oversample_multi(input_json, output_dir, fold, num_class):
    with open(input_json) as f:
        data = json.load(f)

    os_data = []
    folds_data = []
    for i in range(num_class):
        data_fold = [item for item in data if item['target'] == i]
        folds_data.append(data_fold)

    data_folds_length = [len(fold) for fold in folds_data]
    max_len = max(data_folds_length)

    for fold_data in folds_data:
        os_fold_data = list(np.random.choice(fold_data, max_len))
        os_data += os_fold_data

    print('Oversampling dataset size', len(os_data))
    output_fname = fold + '_os.json'
    output_fdir = os.path.join(output_dir, output_fname)
    with open(output_fdir, 'w') as f:
        json.dump(os_data, f)
