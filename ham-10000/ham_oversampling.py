from oversample_json import os_json_multi

fold1 = '/Data/luy8/centermix/ham-10000/json/fold1.json'
fold2 = '/Data/luy8/centermix/ham-10000/json/fold2.json'
fold3 = '/Data/luy8/centermix/ham-10000/json/fold3.json'
fold4 = '/Data/luy8/centermix/ham-10000/json/fold4.json'
fold5 = '/Data/luy8/centermix/ham-10000/json/fold5.json'

output_dir = '/Data/luy8/centermix/ham-10000/json'

os_json_multi(fold1, output_dir, 'fold1', 7)
os_json_multi(fold2, output_dir, 'fold2', 7)
os_json_multi(fold3, output_dir, 'fold3', 7)
os_json_multi(fold4, output_dir, 'fold4', 7)
os_json_multi(fold5, output_dir, 'fold5', 7)

