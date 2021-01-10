import json
import os
from collections import Counter
c5_json = ['renal/json/fold1_us.json',
           'renal/json/fold2_us.json',
           'renal/json/fold3_us.json',
           'renal/json/fold4_us.json',
           'renal/json/fold5_us.json']

c3_label_set = [1, 2, 3]
new_label_encoding = {1:0, 2:1, 3:2}

for file in c5_json:
    with open(file) as f:
        data = json.load(f)

    c3_data = [i for i in data if i['target'] in c3_label_set]
    check = Counter([i['target'] for i in c3_data])
    print(check)

    for img in c3_data:
        img['target'] = new_label_encoding[img['target']]

    check = Counter([i['target'] for i in c3_data])
    print(check)

    root_dir = os.path.dirname(file)
    file_name = os.path.basename(file)
    c3_file = os.path.join(root_dir, 'c3_' + file_name)
    with open(c3_file, 'w') as outf:
        json.dump(c3_data, outf)
