import os
import json
import numpy as np


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

