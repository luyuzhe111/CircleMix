from preprocessing import *

fold1 = 'ham-10000/json/fold1.json'
fold2 = 'ham-10000/json/fold2.json'
fold3 = 'ham-10000/json/fold3.json'
fold4 = 'ham-10000/json/fold4.json'
fold5 = 'ham-10000/json/fold5.json'

output_dir = 'ham-10000/json'

create_train_file(fold1, fold2, fold3, 'trainset1', output_dir)
create_train_file(fold2, fold3, fold4, 'trainset2', output_dir)
create_train_file(fold3, fold4, fold5, 'trainset3', output_dir)
create_train_file(fold4, fold5, fold1, 'trainset4', output_dir)
create_train_file(fold5, fold1, fold2, 'trainset5', output_dir)