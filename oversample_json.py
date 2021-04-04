import os
import json
import numpy as np

fold1 = 'json/fold1.json'
fold2 = 'json/fold2.json'
fold3 = 'json/fold3.json'
fold4 = 'json/fold4.json'
fold5 = 'json/fold5.json'

output_dir = 'json'


def os_json_binary(input_json, output_dir, fold):
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


def os_json_multi(input_json, output_dir, fold, num_class):
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


os_json_binary(fold1, output_dir, 'fold1')
os_json_binary(fold2, output_dir, 'fold2')
os_json_binary(fold3, output_dir, 'fold3')
os_json_binary(fold4, output_dir, 'fold4')
os_json_binary(fold5, output_dir, 'fold5')
