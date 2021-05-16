import pandas as pd
import json
import numpy as np
import os

val_file = 'csv/val_ground_truth.csv'

df = pd.read_csv(val_file)
diag_columns = list(df.columns)[1:]
diag_label = np.argmax(df[diag_columns].values, axis=1)
img_names = df['image'].ravel()

val_data = []
data_root_dir = '/Data/luy8/glomeruli/skin/resized_val'
for img, label in zip(img_names, diag_label):
    val_data.append({
        'image': img + '.png',
        'image_dir': os.path.join(data_root_dir, img + '.png'),
        'label': diag_columns[label],
        'target': int(label)
    })

with open('json/val.json', 'w') as f:
    json.dump(val_data, f)




