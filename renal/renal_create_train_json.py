from create_train_json import create_train_json

fold1 = 'renal/json/ng_fold1.json'
fold2 = 'renal/json/ng_fold2.json'
fold3 = 'renal/json/ng_fold3.json'
fold4 = 'renal/json/ng_fold4.json'
fold5 = 'renal/json/ng_fold5.json'

output_dir = 'renal/json'

create_train_json(fold1, fold2, fold3, 'ng_trainset1', output_dir)
create_train_json(fold2, fold3, fold4, 'ng_trainset2', output_dir)
create_train_json(fold3, fold4, fold5, 'ng_trainset3', output_dir)
create_train_json(fold4, fold5, fold1, 'ng_trainset4', output_dir)
create_train_json(fold5, fold1, fold2, 'ng_trainset5', output_dir)
