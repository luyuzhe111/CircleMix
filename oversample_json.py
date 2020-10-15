import os
import json
import numpy as np

fold1 = 'json/fold1.json'
fold2 = 'json/fold2.json'
fold3 = 'json/fold3.json'
fold4 = 'json/fold4.json'
fold5 = 'json/fold5.json'

output_dir = 'json'


def os_json(input_json, output_dir, fold):
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


os_json(fold1, output_dir, 'fold1')
os_json(fold2, output_dir, 'fold2')
os_json(fold3, output_dir, 'fold3')
os_json(fold4, output_dir, 'fold4')
os_json(fold5, output_dir, 'fold5')
