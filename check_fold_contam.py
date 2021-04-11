import json

def check_contam(fold_dir, dataset):
    fold_lst = []
    num_folds = 5
    for i in range(num_folds):
        with open(f'{fold_dir}/fold{i + 1}.json') as f:
            data = json.load(f)
            patient = [i['patient'] for i in data]
            fold_lst.append(patient)


    print(f'Checking {dataset} dataset...')
    for i in range(num_folds):
        for j in range(i + 1, num_folds):
            cur_fold = set(fold_lst[i])
            next_fold = set(fold_lst[j])
            contam = cur_fold.intersection(next_fold)
            print(f'Between {i+1} and {j+1} contam:', contam)