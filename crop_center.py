from PIL import Image
import os


def main():
    raw_train_dir = '/Data/luy8/centermix/raw_data/train'
    raw_test_dir = '/Data/luy8/centermix/raw_data/test'

    resized_train_dir = '/Data/luy8/centermix/resized_data/train'
    resized_test_dir = '/Data/luy8/centermix/resized_data/test'

    new_size = 256

    crop_center(raw_train_dir, resized_train_dir, new_size)
    crop_center(raw_test_dir, resized_test_dir, new_size)


def crop_center(input_dir, output_dir, new_size):
    for image in os.listdir(input_dir):
        img = Image.open(os.path.join(input_dir, image))
        width, height = img.size
        crop_size = min(width, height)

        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2

        cropped_img = img.crop((left, top, right, bottom))
        resized_img = cropped_img.resize((new_size, new_size))

        output_img_dir = os.path.join(output_dir, image.split('.')[0] + '.png')
        resized_img.save(output_img_dir)


if __name__ == '__main__':
    main()
