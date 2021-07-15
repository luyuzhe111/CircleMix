import json
import os


def extract_label(data_dir, label):
    imgs = os.listdir(data_dir)
    records = []
    for img in imgs:
        records.append(
            {
                'image': img,
                'image_dir': os.path.join(data_dir, img),
                'label': label,
                'target': 0 if label == 'normal' else 1
            }
        )
    return records


if __name__ == "__main__":
    norm_dir = os.path.abspath('test_data/normal')
    scl_dir = os.path.abspath('test_data/sclerosed')
    normal = extract_label(norm_dir, 'normal')
    scl = extract_label(scl_dir, 'sclerosed')

    all = normal + scl

    with open('json/test.json', 'w') as f:
        json.dump(all, f)