from collections import Counter
import json
import os

def data_summary(data_file):
    with open(data_file) as f:
        data = json.load(f)
    print(Counter([i['target'] for i in data]))
    print(len(set([i['subj'] for i in data])))


def check_contam(fold_dir):
    fold_lst = []
    num_folds = 5
    for i in range(num_folds):
        with open(f'{fold_dir}/fold{i + 1}.json') as f:
            data = json.load(f)
            patient = [i['subj'] for i in data]
            fold_lst.append(patient)

    for i in range(num_folds):
        for j in range(i + 1, num_folds):
            cur_fold = set(fold_lst[i])
            next_fold = set(fold_lst[j])
            contam = cur_fold.intersection(next_fold)
            print(f'Between {i+1} and {j+1} contam:', contam)


def check_folds(fold_dir):
    folds = [f'{fold_dir}/fold{i}.json' for i in range(1, 6)]

    for fold in folds:
        with open(fold) as f:
            data = json.load(f)
        labels = [i['target'] for i in data]
        images = [i['image'] for i in data]

        assert len(images) == len(set(images))
        print('##########', os.path.basename(fold), '##########')
        labels_dist = Counter(labels)
        labels_dist = dict(sorted(labels_dist.items(), key=lambda i: i[0]))
        names_set = len(list(set(images)))
        print(labels_dist)
        print("Number of samples: ", (names_set), '\n')


data_summary('json/usable_data.json')
check_contam('json')
check_folds('json')