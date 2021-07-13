import json
import os
import glob
from collections import Counter
import random
import shutil

def get_annot(annot_dir, group, copy_data=False):
    records_lst = []
    records = glob.glob(os.path.join(annot_dir, '*.json'))
    for record in records:
        with open(record) as f:
            data = json.load(f)

        img_name = data['imagePath']
        subj = img_name.split('_')[0]
        disease_dict = data['Glomerular']
        disease_dict.update(data['Bowman'])
        disease_dict.update(data['Other'])

        if all(value is False for value in disease_dict.values()):
            label = 'normal'
            target = 0
            if copy_data:
                shutil.copy(os.path.join(src_dir, img_name),
                            os.path.join('resized_image/by_category/normal', img_name))
        else:
            disease_idx = list(disease_dict.values()).index(True)
            disease = list(disease_dict.keys())[disease_idx]
            if disease == '5.Global obsolescent glomerulosclerosis':
                label = 'obsolescent'
                target = 1
            elif disease == '4.Global solidified glomerulosclerosis':
                label = 'solidified'
                target = 2
            elif disease == '3.Global disappearing glomerulosclerosis':
                label = 'disappearing'
                target = 3
            elif disease == '1.Periglomerular fibrosis':
                label = 'fibrosis'
                target = 4
            else:
                label = 'excluded'
                target = -1

            if label != 'excluded' and copy_data:
                shutil.copy(os.path.join(src_dir, img_name),
                            os.path.join(f'resized_image/by_category/{label}', img_name))

        records_lst.append({
            'subj': subj,
            'image': img_name,
            'image_dir': os.path.abspath(f'resized_image/group{group}/{img_name}'),
            'label': label,
            'target': target
        })

    return records_lst


def get_normal(img_dir, annot_data, group, copy_data=False):
    img = os.listdir(img_dir)
    annot_img = [i['image'] for i in annot_data]
    norm_img = list(set(img) - (set(img).intersection(set(annot_img))))
    records_lst = []
    for i in norm_img:
        if copy_data:
            shutil.copy(os.path.join(img_dir, i), os.path.join('resized_image/by_category/normal', i))
        records_lst.append({
            'subj': i.split('_')[0],
            'image': i,
            'image_dir': os.path.abspath(f'resized_image/group{group}/{i}'),
            'label': 'normal',
            'target': 0
        })

    return records_lst


def get_non_glom(img_dir):
    img = os.listdir(img_dir)
    records_lst = []
    for i in img:
        records_lst.append({
            'subj': i.split('_')[0],
            'image': i,
            'image_dir': os.path.abspath(f"{img_dir}/{i}"),
            'label': 'non_glom',
            'target': 5
        })

    return records_lst


if __name__ == '__main__':
    data = []
    for idx in [1, 2]:
        group_annot_data = get_annot(f'annotation/group{idx}', idx)
        group_norm_data = get_normal(f'resized_image/group{idx}', group_annot_data, idx)

        group_data = [i for i in group_annot_data + group_norm_data if i['target'] != -1]

        if idx == 2:
            group_data = [i for i in group_annot_data if i['target'] not in [0, -1]]

        data.extend(group_data)
        print(Counter([i['label'] for i in group_data]))

    print(Counter([i['label'] for i in data]))

    non_glom = get_non_glom('/Data/luy8/data/renal/no_glom')
    print(len(non_glom))

    data.extend(non_glom)
    for i in data:
        assert os.path.exists(i['image_dir']), i

    with open('json/data.json', 'w') as f:
        json.dump(data, f)

    usable_data = [i for i in data if i['target'] != 4]
    for i in usable_data:
        if i['target'] == 5:
            i['target'] = 4

    print(Counter([i['target'] for i in usable_data]))
    with open('json/usable_data.json', 'w') as f:
        json.dump(usable_data, f)