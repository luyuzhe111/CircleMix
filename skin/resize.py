from utils.preprocessing import crop_center

raw_data_dir = 'raw_train'
resized_data_dir = 'resized_data'

new_size = 256
crop_center(raw_data_dir, resized_data_dir, new_size)