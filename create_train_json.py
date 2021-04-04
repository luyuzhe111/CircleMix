import json
import os


def main():
    fold1 = 'json/fold1_us.json'
    fold2 = 'json/fold2_us.json'
    fold3 = 'json/fold3_us.json'
    fold4 = 'json/fold4_us.json'
    fold5 = 'json/fold5_us.json'

    output_dir = '/Data/luy8/centermix/json'
    create_train_json(fold1, fold2, fold3, 'trainset1_us', output_dir)
    create_train_json(fold2, fold3, fold4, 'trainset2_us', output_dir)
    create_train_json(fold3, fold4, fold5, 'trainset3_us', output_dir)
    create_train_json(fold4, fold5, fold1, 'trainset4_us', output_dir)
    create_train_json(fold5, fold1, fold2, 'trainset5_us', output_dir)


def create_train_json(fold_1, fold_2, fold_3, train_set, output_dir):
    data_set = []
    with open(fold_1, 'r') as json_file:
        data1 = json.load(json_file)
        data_set += data1

    with open(fold_2, 'r') as json_file:
        data2 = json.load(json_file)
        data_set += data2

    with open(fold_3, 'r') as json_file:
        data3 = json.load(json_file)
        data_set += data3

    fname = train_set + '.json'
    with open(os.path.join(output_dir, fname), 'w') as output_file:
        json.dump(data_set, output_file)


if __name__ == '__main__':
    main()
