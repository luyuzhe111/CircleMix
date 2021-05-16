from preprocessing import crop_center

raw_train_dir = 'raw_test'
resized_train_dir = 'resized_test'

new_size = 256
crop_center(raw_train_dir, resized_train_dir, new_size)