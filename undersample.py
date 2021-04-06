import os
import json
import numpy as np


def os_json(input_json, output_dir, fold):
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


def us_json_multi(input_json, output_dir, fold, num_class):
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


def main():
    fold1 = 'json/fold1.json'
    fold2 = 'json/fold2.json'
    fold3 = 'json/fold3.json'
    fold4 = 'json/fold4.json'
    fold5 = 'json/fold5.json'

    output_dir = 'json'

    os_json(fold1, output_dir, 'fold1')
    os_json(fold2, output_dir, 'fold2')
    os_json(fold3, output_dir, 'fold3')
    os_json(fold4, output_dir, 'fold4')
    os_json(fold5, output_dir, 'fold5')
