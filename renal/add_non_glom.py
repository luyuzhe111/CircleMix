import json
import random
from collections import Counter
from preprocessing import create_train_file


def split_fold(root_dir):
    with open('renal/json/non_glom/non_glom.json') as f:
        data = json.load(f)

    subj_set = list(sorted(set([i['subj'] for i in data])))
    num_subj = len(subj_set)

    random.seed(10000)
    fold_assignment = [random.randint(1, 6) for _ in range(num_subj)]
    for item in data:
        for subj, fold in zip(subj_set, fold_assignment):
            if item['subj'] == subj:
                item['fold'] = fold

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

def combine_trainset():
    glom_dir = 'renal/json/all'
    non_glom_dir = 'renal/json/non_glom'

    for i in range(1, 6):
        with open(f'{glom_dir}/trainset{i}.json') as f:
            glom = json.load(f)

        with open(f'{non_glom_dir}/trainset{i}.json') as f:
            non_glom = json.load(f)

        all = glom + non_glom

        with open(f'{glom_dir}/noisy_trainset{i}.json', 'w') as f:
            json.dump(all, f)


if __name__ == "__main__":
    root_dir = 'renal/json/non_glom'
    split_fold(root_dir)
    create_trainset(root_dir)
    combine_trainset()