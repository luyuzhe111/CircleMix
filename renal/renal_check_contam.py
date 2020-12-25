import json
import os

train = '/Data/luy8/centermix/renal/json/trainset1_us.json'
test = '/Data/luy8/centermix/renal/json/fold5_us.json'

with open(train) as ftr:
    train_data = json.load(ftr)

with open(test) as fte:
    test_data = json.load(fte)


train_data = set([i['name'] for i in train_data])

test_data = set([i['name'] for i in test_data])

contamin = train_data.intersection(test_data)

print(contamin)