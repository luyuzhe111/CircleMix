import json
import os
import glob
import shutil
from collections import Counter


root = '/Data/luy8/data/renal/glom'

def copy_data(root_dir):
    dst_record = os.path.join(root_dir, 'record')
    dst_img = os.path.join(root_dir, 'image')
    folders = [i for i in os.listdir(root_dir) if 'gloms' in i]

    for folder in folders:
        folder_dir = os.path.join(root_dir, folder)
        records = glob.glob(os.path.join(folder_dir, '*.json'))
        for record in records:
            shutil.copy(record, dst_record)

        imgs = glob.glob(os.path.join(folder_dir, '*.png'))
        for img in imgs:
            shutil.copy(img, dst_img)


img_dir = os.path.join(root, 'image')
record_dir = os.path.join(root, 'record')

records = glob.glob(os.path.join(record_dir, '*.json'))
records_dict = {}
records_lst = []
norm_count = 0
double_count = 0
record_count = 0



for i in range(len(records)):
    record = records[i]
    print(record)
    print(i + 1)
    with open(record) as f:
        data = json.load(f)

    img_name = data['imagePath']
    subj = img_name.split('_')[0]
    disease_dict = data['Glomerular']

    try:
        disease_idx = list(disease_dict.values()).index(True)
    except ValueError:
        disease_idx = -1
        norm_count += 1
        print(norm_count)
        continue

    if (len([disease_idx])) > 1:
        print('more than 2 labels')
        double_count += 1
        continue

    disease = list(disease_dict.keys())[disease_idx]

    records_lst.append(disease)

    print(i + 1, len(records_lst))

    if subj in records_dict.keys():
        records_dict[subj][disease] = records_dict.get(disease, 0) + 1
    else:
        records_dict[subj] = {}
        records_dict[subj][disease] = 1

# print(records_dict.items())
print(len(records_lst), Counter(records_lst))
print(norm_count, double_count)


old_dir = 'renal/json'
data = []
for i in range(1, 6):
    fold = os.path.join(old_dir, f'fold{i}.json')
    with open(fold) as f:
        data.extend(json.load(f))

print(Counter([i['target'] for i in data]))



