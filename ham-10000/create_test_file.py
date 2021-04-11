import os
import json

test_img_dir = 'ham-10000/resized_test'
test_imgs = os.listdir(test_img_dir)
test_imgs.sort()

test_list = []
for i in test_imgs:
    item = {'name': i.split('.')[0], 'image_dir': os.path.join(test_img_dir, i), 'target':-1}
    test_list.append(item)

with open('json/test.json', 'w') as f:
    json.dump(test_list, f)



