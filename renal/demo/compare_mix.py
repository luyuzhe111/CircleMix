import cv2
import numpy as np

image1 = 'normal.png'
image2 = 'disappearing.png'

img1 = cv2.imread(image1)
img2 = cv2.imread(image2)


def cutmix(img1, img2):
    lam = 0.5
    bbx1, bby1, bbx2, bby2 = rand_bbox(img1.shape, lam)
    img1[bbx1:bbx2, bby1:bby2, :] = img2[bbx1:bbx2, bby1:bby2, :]
    cv2.imshow('img1', img1)
    cv2.waitKey(0)


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def circlemix(img1, img2):
    start = 50
    end = 122

    vertices = polygon_vertices(256, start, end)

    mask = np.zeros((256, 256), np.uint8)

    roi = cv2.fillPoly(mask, np.array([vertices]), 255)

    roi_ch3 = np.repeat(roi[:, :, np.newaxis], 3, axis=2)

    img1[roi_ch3 > 0] = img2[roi_ch3 > 0]
    cv2.imshow('img1', img1)
    cv2.waitKey(0)


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


# cutmix(img1, img2)
circlemix(img1, img2)

cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()
