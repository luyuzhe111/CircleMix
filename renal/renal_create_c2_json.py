import json
import os

import json
import os
from collections import Counter

c3_json = ['renal/json/c3_fold1_us.json',
           'renal/json/c3_fold2_us.json',
           'renal/json/c3_fold3_us.json',
           'renal/json/c3_fold4_us.json',
           'renal/json/c3_fold5_us.json']

c2_label_set = [1, 2]
new_label_encoding = {1:0, 2:1}

for file in c3_json:
    with open(file) as f:
        data = json.load(f)

    c2_data = [i for i in data if i['target'] in c2_label_set]
    check = Counter([i['target'] for i in c2_data])
    print(check)

    for img in c2_data:
        img['target'] = new_label_encoding[img['target']]

    check = Counter([i['target'] for i in c2_data])
    print(check)

    root_dir = os.path.dirname(file)
    file_name = os.path.basename(file)
    c2_file = os.path.join(root_dir, 'c2_' + file_name.split('c3_')[1])
    with open(c2_file, 'w') as outf:
        json.dump(c2_data, outf)
