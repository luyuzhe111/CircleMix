import json
import os

folds = ['renal/json/fold1_us.json',
         'renal/json/fold2_us.json',
         'renal/json/fold3_us.json',
         'renal/json/fold4_us.json',
         'renal/json/fold5_us.json']

for fold in folds:
    with open(fold) as f:
        data = json.load(f)

    binary_mapper = {0:0, 1:1, 2:1, 3:1, 4:0}
    for sample in data:
        sample['target'] = binary_mapper[sample['target']]

    out_file = os.path.join('json', 'binary_' + os.path.basename(fold))
    with open(out_file, 'w') as out_f:
        json.dump(data, out_f)
