from torch.utils import data
import json
from PIL import Image

class DataLoader(data.Dataset):
    def __init__(self, data_list, transform=None):
        with open(data_list) as json_file:
            data = json.load(json_file)

        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        image_dir = data['image_dir']
        label = data['target']

        img = Image.open(image_dir)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, image_dir


    def __len__(self):
        return len(self.data)