import json
import random

fold1 = '/Data/luy8/centermix/renal/json/c3_fold1_us.json'
fold2 = '/Data/luy8/centermix/renal/json/c3_fold2_us.json'
fold3 = '/Data/luy8/centermix/renal/json/c3_fold3_us.json'
fold4 = '/Data/luy8/centermix/renal/json/c3_fold4_us.json'
fold5 = '/Data/luy8/centermix/renal/json/c3_fold5_us.json'

with open(fold5) as f:
    data = json.load(f)

label0 = [i for i in data if i['target']==0 ]
label1 = [i for i in data if i['target']==1 ]
label2 = [i for i in data if i['target']==2 ]
# label3 = [i for i in data if i['target']==3 ]
# label4 = [i for i in data if i['target']==4 ]

random.shuffle(label0)
label0 = label0[:50]
# label3 = list(np.random.choice(label3, len(label3) - 50))
# label0 = list(np.random.choice(label0, len(label0) - 600))
# label1 = list(np.random.choice(label1, len(label1) - 100))

new_data = label0 + label1 + label2
print(len(new_data))
print(len(label0), len(label1), len(label2))

with open('/Data/luy8/centermix/renal/json/c3_fold5_us_us.json', 'w') as f:
    json.dump(new_data, f)