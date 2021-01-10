import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import torch
import json
import imgaug.augmenters as iaa


class DataLoader(data.Dataset):
    def __init__(self, train_list, transform=None, split=None, aug=None):
        with open(train_list) as json_file:
            data = json.load(json_file)

        self.data = data
        self.train_list = train_list
        self.transform = transform
        self.split = split
        self.target_transform = None
        self.aug = aug

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        data = self.data[index]
        image_dir = data['image_dir']
        label = data['target']

        img = cv2.imread(image_dir)

        if self.split == 'train':
            if self.aug == 'True':
                img = self.augmentation(normalise(img))
            else:
                img = normalise(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, label, image_dir

        elif self.split == 'val':
            img = normalise(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, label, image_dir

    def augmentation(self, pil_img):
        input_img = np.expand_dims(pil_img, axis=0)

        if self.split == 'train':
            seq = iaa.Sequential([
                iaa.Crop(px=(0, 16)),
                iaa.Fliplr(0.5),  # horizontal flips
                iaa.Flipud(0.5),  # vertical  flips
            ], random_order=True)  # apply augmenters in random order
            images_aug = seq(images=input_img)

            # if we would like to see the data augmentation
            # seq.show_grid(images_aug[0], cols=8, rows=8)
            return images_aug[0]

        return pil_img

    def __len__(self):
        return len(self.data)


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='HWC', target='CHW'):
    return x.transpose([source.index(d) for d in target])


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """

    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img = np.transpose(img, [1, 2, 0])
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        img = np.transpose(img, [2, 0, 1])
        return img
