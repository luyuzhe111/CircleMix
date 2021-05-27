from torch.utils import data
import json
from PIL import Image
from imgaug import augmenters as iaa
import cv2
import numpy as np
import torch
import torch.nn as nn


class GaussianBlur(torch.nn.Module):
    def forward(self, img):
        img = np.array(img)
        img = np.expand_dims(img, axis=0)

        seq = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 3.0))
        ])

        img = seq(images=img)

        return Image.fromarray(img[0])


