import json
import random
import numpy as np

fold1 = '/Data/luy8/centermix/renal/json/fold1.json'
fold2 = '/Data/luy8/centermix/renal/json/fold2.json'
fold3 = '/Data/luy8/centermix/renal/json/fold3.json'
fold4 = '/Data/luy8/centermix/renal/json/fold4.json'
fold5 = '/Data/luy8/centermix/renal/json/fold5.json'

with open(fold5) as f:
    data = json.load(f)

label0 = [i for i in data if i['target']==0 ]
label1 = [i for i in data if i['target']==1 ]
label2 = [i for i in data if i['target']==2 ]
label3 = [i for i in data if i['target']==3 ]
label4 = [i for i in data if i['target']==4 ]

label4 = list(np.random.choice(label4, len(label4) - 400))
# label3 = list(np.random.choice(label3, len(label3) - 50))
# label0 = list(np.random.choice(label0, len(label0) - 600))
# label1 = list(np.random.choice(label1, len(label1) - 100))

new_data = label0 + label1 + label2 + label3 + label4
print(len(new_data))
print(len(label0), len(label1), len(label2), len(label3), len(label4))

with open('/Data/luy8/centermix/renal/json/fold5_us.json', 'w') as f:
    json.dump(new_data, f)