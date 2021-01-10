import os
import json
import pandas as pd


def main():
    train_csv = '/Data/luy8/centermix/train.csv'
    df = pd.read_csv(train_csv)

    root_dir = '/Data/luy8/centermix'
    data_dir = '/Data/luy8/centermix/resized_data/train'
    data_list = []
    for index, row in df.iterrows():
        img = row['image_name']
        img_dir = os.path.join(data_dir, img + '.png')
        target = row['target']

        one_entry = {'name': img, 'image_dir': img_dir, 'target': target}
        data_list.append(one_entry)

    output_json = os.path.join(root_dir, 'train.json')
    with open(output_json, 'w') as out_file:
        json.dump(data_list, out_file)


if __name__ == '__main__':
    main()
