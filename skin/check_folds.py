import os
import json
from collections import Counter


folds = [f'json/fold{i}.json' for i in range(1, 6)]

for fold in folds:
    with open(fold) as f:
        data = json.load(f)
    labels = [i['target'] for i in data]
    images = [i['image'] for i in data]

    assert len(images) == len(set(images))
    print('##########', os.path.basename(fold), '##########')
    labels_dist = Counter(labels)
    labels_dist = dict(sorted(labels_dist.items(), key=lambda i:i[0]))
    names_set = len(list(set(images)))
    print(labels_dist)
    print("Number of samples: ", (names_set), '\n')