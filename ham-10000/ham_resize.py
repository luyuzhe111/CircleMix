from crop_center import crop_center

raw_train_dir = '/Data/luy8/centermix/ham-10000/raw_data'
resized_train_dir = '/Data/luy8/centermix/ham-10000/resized_data'

new_size = 256
crop_center(raw_train_dir, resized_train_dir, new_size)