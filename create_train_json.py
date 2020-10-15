import json
import os

fold1 = 'json/fold1_os.json'
fold2 = 'json/fold2_os.json'
fold3 = 'json/fold3_os.json'
fold4 = 'json/fold4_os.json'
fold5 = 'json/fold5_os.json'


def create_train_json(fold1, fold2, fold3, train_set):
    data_set = []
    with open(fold1, 'r') as json_file:
        data1 = json.load(json_file)
        data_set += data1

    with open(fold2, 'r') as json_file:
        data2 = json.load(json_file)
        data_set += data2

    with open(fold3, 'r') as json_file:
        data3 = json.load(json_file)
        data_set += data3

    fname = train_set + '.json'
    with open(os.path.join('json', fname), 'w') as output_file:
        json.dump(data_set, output_file)


create_train_json(fold1, fold2, fold3, 'trainset1_os')
create_train_json(fold2, fold3, fold4, 'trainset2_os')
create_train_json(fold3, fold4, fold5, 'trainset3_os')
create_train_json(fold4, fold5, fold1, 'trainset4_os')
create_train_json(fold5, fold1, fold2, 'trainset5_os')