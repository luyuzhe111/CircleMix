import cv2
import numpy as np
from PIL import Image

image1 = '/Data/luy8/centermix/resized_data/train/ISIC_0015719.png'
image2 = '/Data/luy8/centermix/resized_data/train/ISIC_0052212.png'

img1 = cv2.imread(image1)
img2 = cv2.imread(image2)


def polygon_vertices(size, start, end):
    side = size-1
    center = (int(side/2), int(side/2))
    bound1 = coordinate(start, side)
    bound2 = coordinate(end, side)

    square_vertices = {90: (0, side), 180: (side, side), 270: (side, 0)}
    inner_vertices = []
    if start < 90 < end:
        inner_vertices.append(square_vertices[90])
    if start < 180 < end:
        inner_vertices.append(square_vertices[180])
    if start < 270 < end:
        inner_vertices.append(square_vertices[270])

    all_vertices = [center] + [bound1] + inner_vertices + [bound2]
    return all_vertices


def coordinate(num, side):
    length = side + 1
    if 0 <= num < 90:
        return 0, int(num/90*length)
    elif 90 <= num < 180:
        return int((num-90)/90*length), side
    elif 180 <= num < 270:
        return side, side - int((num-180)/90*length)
    elif 270 <= num <= 360:
        return side - int((num - 270)/90*length), 0


vertices = polygon_vertices(256, 30, 90)
print(vertices)

mask = np.zeros((256, 256), np.uint8)

roi = cv2.fillPoly(mask, np.array([vertices]), 255)

roi_ch3 = np.repeat(roi[:, :, np.newaxis], 3, axis=2)

test = img1.copy()
cv2.fillPoly(test, np.array([vertices]), 255)

cv2.imshow('test', test)
cv2.waitKey(0)

