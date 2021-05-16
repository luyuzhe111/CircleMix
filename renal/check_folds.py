from collections import Counter
import json
import os

folds = [f'json/all/fold{i}.json' for i in range(1, 6)]

for fold in folds:
    with open(fold) as f:
        data = json.load(f)
    labels = [i['label'] for i in data]
    images = [i['image'] for i in data]

    assert len(images) == len(set(images))
    print('##########', os.path.basename(fold), '##########')
    labels_dist = Counter(labels)
    labels_dist = dict(sorted(labels_dist.items(), key=lambda i:i[0]))
    names_set = len(list(set(images)))
    print(labels_dist)
    print("Number of samples: ", (names_set), '\n')


group1_dir = 'json/group1'
group1_data = []
for i in range(1, 6):
    fold = os.path.join(group1_dir, f'fold{i}.json')
    with open(fold) as f:
        data = json.load(f)

    group1_data += data

with open('json/group1/data.json', 'w') as f:
    json.dump(group1_data, f)

all_dir = 'json/all'
with open(os.path.join(all_dir, 'data.json')) as f:
    all_data = json.load(f)

group1_norm = set([i['name'] for i in group1_data if i['target'] == 4])
new_norm = set([i['image'] for i in all_data if i['target'] == 4])

print(len(group1_norm), len(new_norm), len(group1_norm.intersection(new_norm)))