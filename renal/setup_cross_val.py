import json
import random
from collections import Counter
from preprocessing import create_train_file

def split_fold(root_dir):
    with open(f'{root_dir}/data.json') as f:
        data = json.load(f)

    subj_set = list(sorted(set([i['subj'] for i in data])))
    num_subj = len(subj_set)

    random.seed(10)
    fold_assignment = [random.randint(1, 6) for _ in range(num_subj)]
    for subj, fold in zip(subj_set, fold_assignment):
        for item in data:
            if subj == item['subj']:
                item['fold'] = fold

    print(Counter([i['subj'] for i in data]))

    for fold in range(1, 6):
        fold_data = [i for i in data if i['fold'] == fold]

        print(sorted(Counter([i['target'] for i in fold_data]).items()))
        with open(f'{root_dir}/fold{fold}.json', 'w') as f:
            json.dump(fold_data, f)


def create_trainset(root_dir):
    for i in range(1, 6):
        fold_i = f'{root_dir}/fold{i}.json'
        j = i + 1
        k = i + 2

        if j > 5:
            j = j % 5

        if k > 5:
            k = k % 5

        fold_j = f'{root_dir}/fold{j}.json'
        fold_k = f'{root_dir}/fold{k}.json'

        print(fold_i, fold_j, fold_k)

        create_train_file(fold_i, fold_j, fold_k, f'trainset{i}', root_dir)


if __name__ == '__main__':
    root_dir = 'json/all'
    split_fold(root_dir)
    create_trainset(root_dir)