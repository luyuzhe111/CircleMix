import json
import random
import numpy as np

fold1 = '/Data/luy8/centermix/renal/json/fold1.json'
fold2 = '/Data/luy8/centermix/renal/json/fold2.json'
fold3 = '/Data/luy8/centermix/renal/json/fold3.json'
fold4 = '/Data/luy8/centermix/renal/json/fold4.json'
fold5 = '/Data/luy8/centermix/renal/json/fold5.json'

# fold1
with open(fold1) as f:
    data = json.load(f)

label0 = [i for i in data if i['target']==0 ]
label1 = [i for i in data if i['target']==1 ]
label2 = [i for i in data if i['target']==2 ]
label3 = [i for i in data if i['target']==3 ]
label4 = [i for i in data if i['target']==4 ]

random.shuffle(label4)
label4 = list(np.random.choice(label4, len(label4) - 400, replace=False))

new_data = label0 + label1 + label2 + label3 + label4
print(len(new_data))
print(len(label0), len(label1), len(label2), len(label3), len(label4))

with open('/Data/luy8/centermix/renal/json/fold1_trim.json', 'w') as f:
    json.dump(new_data, f)

# trim fold 2
with open(fold2) as f:
    data = json.load(f)

label0 = [i for i in data if i['target']==0 ]
label1 = [i for i in data if i['target']==1 ]
label2 = [i for i in data if i['target']==2 ]
label3 = [i for i in data if i['target']==3 ]
label4 = [i for i in data if i['target']==4 ]

random.shuffle(label1)
label1 = list(np.random.choice(label1, len(label1) - 400, replace=False))

new_data = label0 + label1 + label2 + label3 + label4
print(len(new_data))
print(len(label0), len(label1), len(label2), len(label3), len(label4))

with open('/Data/luy8/centermix/renal/json/fold2_trim.json', 'w') as f:
    json.dump(new_data, f)

# fold3
with open(fold3) as f:
    data = json.load(f)

label0 = [i for i in data if i['target']==0 ]
label1 = [i for i in data if i['target']==1 ]
label2 = [i for i in data if i['target']==2 ]
label3 = [i for i in data if i['target']==3 ]
label4 = [i for i in data if i['target']==4 ]

random.shuffle(label3)
label0 = list(np.random.choice(label0, len(label0) - 200, replace=False))

new_data = label0 + label1 + label2 + label3 + label4
print(len(new_data))
print(len(label0), len(label1), len(label2), len(label3), len(label4))

with open('/Data/luy8/centermix/renal/json/fold3_trim.json', 'w') as f:
    json.dump(new_data, f)

#fold4
with open(fold4) as f:
    data = json.load(f)

label0 = [i for i in data if i['target']==0 ]
label1 = [i for i in data if i['target']==1 ]
label2 = [i for i in data if i['target']==2 ]
label3 = [i for i in data if i['target']==3 ]
label4 = [i for i in data if i['target']==4 ]

random.shuffle(label0)
label0 = list(np.random.choice(label0, len(label0) - 600, replace=False))

new_data = label0 + label1 + label2 + label3 + label4
print(len(new_data))
print(len(label0), len(label1), len(label2), len(label3), len(label4))

with open('/Data/luy8/centermix/renal/json/fold4_trim.json', 'w') as f:
    json.dump(new_data, f)

#fold 5
with open(fold5) as f:
    data = json.load(f)

label0 = [i for i in data if i['target']==0 ]
label1 = [i for i in data if i['target']==1 ]
label2 = [i for i in data if i['target']==2 ]
label3 = [i for i in data if i['target']==3 ]
label4 = [i for i in data if i['target']==4 ]

random.shuffle(label4)
label4 = list(np.random.choice(label4, len(label4) - 380, replace=False))

new_data = label0 + label1 + label2 + label3 + label4
print(len(new_data))
print(len(label0), len(label1), len(label2), len(label3), len(label4))

with open('/Data/luy8/centermix/renal/json/fold5_trim.json', 'w') as f:
    json.dump(new_data, f)