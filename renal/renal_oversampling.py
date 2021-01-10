from oversample_json import os_json_multi

fold1 = '/Data/luy8/centermix/renal/json/fold1_us.json'
fold2 = '/Data/luy8/centermix/renal/json/fold2_us.json'
fold3 = '/Data/luy8/centermix/renal/json/fold3_us.json'
fold4 = '/Data/luy8/centermix/renal/json/fold4_us.json'
fold5 = '/Data/luy8/centermix/renal/json/fold5_us.json'

output_dir = '/Data/luy8/centermix/renal/json'

os_json_multi(fold1, output_dir, 'fold1_us', 5)
os_json_multi(fold2, output_dir, 'fold2_us', 5)
os_json_multi(fold3, output_dir, 'fold3_us', 5)
os_json_multi(fold4, output_dir, 'fold4_us', 5)
os_json_multi(fold5, output_dir, 'fold5_us', 5)

