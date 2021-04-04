import cv2
import imgaug.augmenters as iaa
import numpy as np

img_dir = 'resized_image/22558_2017-04-08 12_34_57-x-ROI_0-x-glomerulus-x-50195-x-80396-x-1082-x-1083.png'
img = cv2.imread(img_dir)

img = np.expand_dims(img, axis=0)

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)),
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Flipud(0.5),  # vertical flips
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-8, 8)
    )
], random_order=True)  # apply augmenters in random order

images_aug = seq(images=img)

seq.show_grid(images_aug[0], cols=2, rows=2)
