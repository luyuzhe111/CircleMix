import json
import random
from collections import Counter
from preprocessing import create_train_file
import pandas as pd


def split_fold(root_dir):
    with open(f'{root_dir}/data.json') as f:
        data = json.load(f)

    subj_set = list(sorted(set([i['subj'] for i in data])))
    num_subj = len(subj_set)

    subj_records = []
    for subj in subj_set:
        subj_data = [i for i in data if i['subj'] == subj]
        prefix = subj_data[0]['image']
        subj_label = [i['target'] for i in subj_data]

        counter = Counter(subj_label)
        labels = ['normal', 'obsolescent', 'solidified', 'disappearing', 'fibrosis']
        subj_record = [subj, prefix]
        for idx, _ in enumerate(labels):
            subj_record.append(counter[idx])
        subj_records.append(subj_record)

    df = pd.DataFrame(subj_records, columns=['subj', 'prefix', 'normal', 'obsolescent', 'solidified', 'disappearing', 'fibrosis'])
    df.to_csv('csv/subj_summary.csv')

    # 10 is the best
    random.seed(0)
    fold_assignment = [random.randint(1, 6) for _ in range(num_subj)]
    for item in data:
        for subj, fold in zip(subj_set, fold_assignment):
            if item['subj'] == subj:
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