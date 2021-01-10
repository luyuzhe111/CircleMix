import json
import os
from collections import Counter

folds_json = ['renal/json/fold1.json',
              'renal/json/fold2.json',
              'renal/json/fold3.json',
              'renal/json/fold4.json',
              'renal/json/fold5.json']

ng_label_set = [0, 4]
new_label_encoding = {0:0, 4:1}

for file in folds_json:
    with open(file) as f:
        data = json.load(f)

    ng_data = [i for i in data if i['target'] in ng_label_set]
    check = Counter([i['target'] for i in ng_data])
    print(check)

    for img in ng_data:
        img['target'] = new_label_encoding[img['target']]

    check = Counter([i['target'] for i in ng_data])
    print(check)

    root_dir = os.path.dirname(file)
    file_name = os.path.basename(file)
    ng_file = os.path.join(root_dir, 'ng_' + file_name)
    with open(ng_file, 'w') as outf:
        json.dump(ng_data, outf)
