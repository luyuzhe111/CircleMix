import json

fold1 = 'renal/json/fold1.json'
fold2 = 'renal/json/fold2.json'
fold3 = 'renal/json/fold3.json'
fold4 = 'renal/json/fold4.json'
fold5 = 'renal/json/fold5.json'

folds = [fold1, fold2, fold3, fold4, fold5]

for fold in folds:
    with open(fold) as f:
        data = json.load(f)