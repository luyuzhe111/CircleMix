from create_train_json import create_train_json

fold1 = '/Data/luy8/centermix/ham-10000/json/fold1.json'
fold2 = '/Data/luy8/centermix/ham-10000/json/fold2.json'
fold3 = '/Data/luy8/centermix/ham-10000/json/fold3.json'
fold4 = '/Data/luy8/centermix/ham-10000/json/fold4.json'
fold5 = '/Data/luy8/centermix/ham-10000/json/fold5.json'

output_dir = '/Data/luy8/centermix/ham-10000/json'

create_train_json(fold1, fold2, fold3, 'trainset1', output_dir)
create_train_json(fold2, fold3, fold4, 'trainset2', output_dir)
create_train_json(fold3, fold4, fold5, 'trainset3', output_dir)
create_train_json(fold4, fold5, fold1, 'trainset4', output_dir)
create_train_json(fold5, fold1, fold2, 'trainset5', output_dir)