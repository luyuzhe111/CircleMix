from collections import Counter
import json
import os

fold1 = 'renal/json/ng_fold1.json'
fold2 = 'renal/json/ng_fold2.json'
fold3 = 'renal/json/ng_fold3.json'
fold4 = 'renal/json/ng_fold4.json'
fold5 = 'renal/json/ng_fold5.json'

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
    print("Number of unique samples: ", (names_set))
    print()