from collections import Counter
import json
import os

fold1 = 'renal/json/fold1_trim.json'
fold2 = 'renal/json/fold2_trim.json'
fold3 = 'renal/json/fold3_trim.json'
fold4 = 'renal/json/fold4_trim.json'
fold5 = 'renal/json/fold5_trim.json'

folds = [fold1, fold2, fold3, fold4, fold5]

for fold in folds:
    with open(fold) as f:
        data = json.load(f)
    labels = [i['target'] for i in data]
    images = [i['name'] for i in data]
    print('##########', os.path.basename(fold), '##########')
    labels_dist = Counter(labels)
    labels_dist = dict(sorted(labels_dist.items(), key=lambda i:i[0]))
    names_set = len(list(set(images)))
    print(labels_dist)
    print("Number of samples: ", (names_set))
    print()