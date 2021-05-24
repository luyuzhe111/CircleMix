import json
import os
import glob
from collections import Counter
import random
import shutil

def get_annot(annot_dir, group):
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

        # src_dir = f'renal/resized_image/group{group}'
        if all(value is False for value in disease_dict.values()):
            label = 'normal'
            target = 0
            # shutil.copy(os.path.join(src_dir, img_name),
            #             os.path.join('renal/resized_image/by_category/normal', img_name))
        else:
            disease_idx = list(disease_dict.values()).index(True)
            disease = list(disease_dict.keys())[disease_idx]
            if disease == '5.Global obsolescent glomerulosclerosis':
                label = 'obsolescent'
                target = 1
                # shutil.copy(os.path.join(src_dir, img_name),
                #             os.path.join('renal/resized_image/by_category/obsolescent', img_name))
            elif disease == '4.Global solidified glomerulosclerosis':
                label = 'solidified'
                target = 2
                # shutil.copy(os.path.join(src_dir, img_name),
                #             os.path.join('renal/resized_image/by_category/solidified', img_name))
            elif disease == '3.Global disappearing glomerulosclerosis':
                label = 'disappearing'
                target = 3
                # shutil.copy(os.path.join(src_dir, img_name),
                #             os.path.join('renal/resized_image/by_category/disappearing', img_name))
            elif disease == '1.Periglomerular fibrosis':
                label = 'fibrosis'
                target = 4
                # shutil.copy(os.path.join(src_dir, img_name),
                #             os.path.join('renal/resized_image/by_category/fibrosis', img_name))
            else:
                label = 'excluded'
                target = -1

        records_lst.append({
            'subj': subj,
            'image': img_name,
            'image_dir': os.path.join(f'/Data/luy8/glomeruli/renal/resized_image/group{group}', img_name),
            'label': label,
            'target': target
        })

    return records_lst


def get_normal(img_dir, annot_data, group):
    img = os.listdir(img_dir)
    annot_img = [i['image'] for i in annot_data]
    norm_img = list(set(img) - (set(img).intersection(set(annot_img))))
    records_lst = []
    for i in norm_img:
        shutil.copy(os.path.join(img_dir, i), os.path.join('renal/resized_image/by_category/normal', i))
        records_lst.append({
            'subj': i.split('_')[0],
            'image': i,
            'image_dir': os.path.join(f'/Data/luy8/glomeruli/renal/resized_image/group{group}', i),
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
            'image_dir': os.path.join(img_dir, i),
            'label': 'non_glom',
            'target': 5
        })

    return records_lst


if __name__ == '__main__':
    # data = []
    # for idx in [1, 2]:
    #     group_annot_data = get_annot(f'renal/annotation/group{idx}', idx)
    #     group_norm_data = get_normal(f'renal/resized_image/group{idx}', group_annot_data, idx)
    #
    #     group_data = [i for i in group_annot_data + group_norm_data if i['target'] != -1]
    #     data.extend(group_data)
    #     print(Counter([i['label'] for i in group_data]))
    #
    # obs_data = [i for i in data if i['label'] == 'obsolescent']
    # other_data = [i for i in data if i['label'] != 'obsolescent']
    #
    # random.seed(0)
    # random.shuffle(obs_data)
    # trimmed_data = obs_data + other_data
    #
    # print(Counter([i['label'] for i in trimmed_data]))
    # with open('renal/json/all/data.json', 'w') as f:
    #     json.dump(trimmed_data, f)

    non_glom = get_non_glom('/Data/luy8/data/renal/no_glom')
    with open('renal/json/non_glom/non_glom.json', 'w') as f:
        json.dump(non_glom, f)