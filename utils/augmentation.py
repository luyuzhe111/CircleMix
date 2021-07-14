from torch.utils import data
import json
import math
from PIL import Image
from imgaug import augmenters as iaa
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List, Optional
from collections.abc import Sequence
import numbers
import random

class GaussianBlur(torch.nn.Module):
    def forward(self, img):
        img = np.array(img)
        img = np.expand_dims(img, axis=0)

        seq = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 3.0))
        ])

        img = seq(images=img)

        return Image.fromarray(img[0])


class RandomErasing(torch.nn.Module):
    """ Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.ToTensor(),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(
            img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1, )).item()
            j = torch.randint(0, img_w - w + 1, size=(1, )).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img


    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    "{} (number of input channels)".format(img.shape[-3])
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return erase(img, x, y, h, w, v, self.inplace)
        return img


    def __repr__(self):
        s = '(p={}, '.format(self.p)
        s += 'scale={}, '.format(self.scale)
        s += 'ratio={}, '.format(self.ratio)
        s += 'value={}, '.format(self.value)
        s += 'inplace={})'.format(self.inplace)
        return self.__class__.__name__ + s


def erase(img: Tensor, i: int, j: int, h: int, w: int, v: Tensor, inplace: bool = False) -> Tensor:
    """ Erase the input Tensor Image with given value.
    This transform does not support PIL Image.
    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.
    Returns:
        Tensor Image: Erased image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.clone()

    img[..., i:i + h, j:j + w] = v
    return img



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


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
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


