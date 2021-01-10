import os
import pandas as pd
import json

csv_dir = 'renal/folds_csv'
csv_f = os.listdir(csv_dir)
csv_f.sort()

json_dir = 'renal/json'

data_dir = 'renal/resized_image'

fold = 1
for csv in csv_f:
    f_dir = os.path.join(csv_dir, csv)
    df = pd.read_csv(f_dir)
    json_f = []
    for row in df.itertuples(index=False):
        image = row[0]
        condition = row[1]
        label = row[2]
        if label != -1:
            json_f.append({'name': image, 'patient':image.split(' ')[0],
                           'image_dir': os.path.join(data_dir, image), 'target': label})

    out_f_dir = os.path.join(json_dir, 'fold'+str(fold)+'.json')
    print(len(json_f))
    with open(out_f_dir, 'w') as jf:
        json.dump(json_f, jf)

    fold += 1

