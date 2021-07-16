import json
import random
from collections import Counter
from utils.preprocessing import save_train_file
import pandas as pd


def split_fold(root_dir):
    with open(f'{root_dir}/usable_data.json') as f:
        data = json.load(f)

    labels = ['normal', 'obsolescent', 'solidified', 'disappearing', 'non-glomerulous']
    num_classes = len(labels)
    subjects = list(sorted(set([i['subj'] for i in data])))
    num_subj = len(subjects)

    subj_records = []
    for subj in subjects:
        subj_data = [i for i in data if i['subj'] == subj]
        prefix = subj_data[0]['image'].split(' ')[0]
        subj_label = [i['target'] for i in subj_data]

        counter = Counter(subj_label)
        subj_record = [subj, prefix] + [counter[i] for i in range(len(labels))]
        subj_records.append(subj_record)

    df = pd.DataFrame(subj_records, columns=['subj', 'prefix',
                                             'normal', 'obsolescent', 'solidified', 'disappearing', 'non-glomerulous'])
    df.to_csv('csv/subj_summary.csv')

    # 40
    random.seed(40)
    fold_assignment = dict(zip(subjects, [random.randint(1, num_classes) for _ in range(num_subj)]))  # randint is inclusive...
    for item in data:
        item['fold'] = fold_assignment[item['subj']]

    print(Counter([i['subj'] for i in data]))

    for fold in range(1, num_classes + 1):
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

        save_train_file(fold_i, fold_j, fold_k, f'trainset{i}', root_dir)


if __name__ == '__main__':
    out_dir = 'json'
    split_fold(out_dir)
    create_trainset(out_dir)