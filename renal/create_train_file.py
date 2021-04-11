from create_trainset import create_trainset

fold1 = 'json/fold1_trim.json'
fold2 = 'json/fold2_trim.json'
fold3 = 'json/fold3_trim.json'
fold4 = 'json/fold4_trim.json'
fold5 = 'json/fold5_trim.json'

output_dir = 'json'

create_trainset(fold1, fold2, fold3, 'trainset1_trim', output_dir)
create_trainset(fold2, fold3, fold4, 'trainset2_trim', output_dir)
create_trainset(fold3, fold4, fold5, 'trainset3_trim', output_dir)
create_trainset(fold4, fold5, fold1, 'trainset4_trim', output_dir)
create_trainset(fold5, fold1, fold2, 'trainset5_trim', output_dir)
