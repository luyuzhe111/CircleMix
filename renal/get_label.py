import json
import os
import glob
from collections import Counter
import random

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

        try:
            disease_idx = list(disease_dict.values()).index(True)
        except ValueError:
            continue

        if (len([disease_idx])) > 1:
            print('more than 2 labels')
            continue

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
            label = 'exlcuded'
            target = -1

        records_lst.append({
            'subj': subj,
            'image': img_name,
            'image_dir': os.path.join(f'../renal/resized_image/group{group}', img_name),
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
        records_lst.append({
            'subj': i.split(' ')[0],
            'image': i,
            'image_dir': os.path.join(f'../renal/resized_image/group{group}', i),
            'label': 'normal',
            'target': 0
        })

    return records_lst


if __name__ == '__main__':
    data = []
    for i in [1, 2]:
        group_annot_data = get_annot(f'renal/annotation/group{i}', i)
        group_norm_data = get_normal(f'renal/resized_image/group{i}', group_annot_data, i)

        group_data = [i for i in group_annot_data + group_norm_data if i['target'] != -1]
        data.extend(group_data)
        print(Counter([i['label'] for i in group_data]))

    # filter out some obsolescent data, specifically subj 22861, which has over 1200 obsolescent glomeruli
    obs_data = [i for i in data if i['label'] == 'obsolescent']
    other_data = [i for i in data if i['label'] != 'obsolescent']

    random.seed(0)
    random.shuffle(obs_data)
    trimmed_data = obs_data[:int(len(obs_data) / 2)] + other_data

    # obs_most = [i for i in obs_data if i['subj'] == '22861']
    # obs_other = [i for i in obs_data if i['subj'] != '22861']

    # random.seed(0)
    # random.shuffle(obs_most)
    # random.shuffle(obs_other)
    #
    # deduction = 600
    # trimmed_obs_data = (obs_most[:len(obs_most) - deduction] +
    #                     obs_other[:int(len(obs_data)/2) - (len(obs_most) - deduction)])
    #
    # trimmed_data = trimmed_obs_data + other_data
    # print(len([i for i in trimmed_data if i['subj'] == '22861']))
    # print(len([i for i in trimmed_data if i['label'] == 'obsolescent']))

    print(Counter([i['label'] for i in trimmed_data]))
    with open('renal/json/all/data.json', 'w') as f:
        json.dump(trimmed_data, f)