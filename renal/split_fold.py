import os
import json
import pandas as pd


def create_sublist(file_dir, fold1, fold2, fold3, fold4, fold5):
    for file in os.listdir(file_dir):
        subject = file.split('_')[0]
        if subject in fold1:
            fold1_list.append(file)
        elif subject in fold2:
            fold2_list.append(file)
        elif subject in fold3:
            fold3_list.append(file)
        elif subject in fold4:
            fold4_list.append(file)
        elif subject in fold5:
            fold5_list.append(file)

    return fold1_list, fold2_list, fold3_list, fold4_list, fold5_list


def output_csv(sublist, list_name):
    hist = dict()
    df = pd.DataFrame(columns=['image', 'condition', 'label'])
    row = 0

    for item in sublist:
        lst = item.split('.')
        json_file = lst[0] + '.json'
        if json_file in os.listdir(json_dir):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path) as json_data:
                data = json.load(json_data)
            condition = None
            for label in labels:
                attributes = data[label]
                for attribute in attributes:
                    if attributes[attribute]:
                        if condition is None:
                            lst = attribute.split('.')
                            condition = lst[1]
                        else:
                            print(json_path, 'More than two attributes')
                            quit()
            if condition is None:
                print(data['imagePath'], 'This image has no condition')
                condition = 'Normal'

            label_num = 0
            if condition == 'Wire loop' or condition == 'Capsular drop':
                hist[condition] = hist.get(condition, 0) + 1
                label_num = -1
            else:
                for i in range(len(conditions)):
                    if conditions[i] == condition:
                        hist[condition] = hist.get(condition, 0) + 1
                        label_num = i

            image = data['imagePath']
            df.loc[row] = [image, condition, label_num]

        else:
            hist[conditions[0]] = hist.get(conditions[0], 0) + 1
            df.loc[row] = [item, conditions[0], 0]

        row = row + 1

    print('========%s=========' % list_name)
    print(hist)

    df.sort_values(by=['image'], ascending=True, inplace=True)
    output_csv_file = os.path.join(output_dir, '%s.csv' % list_name)
    df.to_csv(output_csv_file, index=False)


if __name__ == "__main__":
    image_dir = 'renal/resized_image'
    json_dir = 'renal/annotation_json'

    output_dir = 'folds'

    labels = ['Glomerular', 'Bowman', 'Other']

    conditions = ['Normal',
                  'Global obsolescent glomerulosclerosis',
                  'Global solidified glomerulosclerosis',
                  'Global disappearing glomerulosclerosis',
                  'Periglomerular fibrosis',
                  'Capsular drop',
                  'Wire loop']

    fold_1 = ['22558', '22732', '24738', '22862', '23498', '25119']
    fold_2 = ['22861', '22998', '24739']
    fold_3 = ['22559', '22863', '22899', '23681']
    fold_4 = ['22560', '22859', '22860', '25121']
    fold_5 = ['22900', '22901', '22918', '23418', '23499', '25118']

    fold1_list, fold2_list, fold3_list, fold4_list, fold5_list = create_sublist(image_dir,
                                                                                fold_1,
                                                                                fold_2,
                                                                                fold_3,
                                                                                fold_4,
                                                                                fold_5,)

    output_csv(fold1_list, 'fold1')
    output_csv(fold2_list, 'fold2')
    output_csv(fold3_list, 'fold3')
    output_csv(fold4_list, 'fold4')
    output_csv(fold5_list, 'fold5')



