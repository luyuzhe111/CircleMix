import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import json

from data_loader import DataLoader
from util import parse_args, create_model, load_checkpoint

args = parse_args('filter')
checkpoint = os.path.join(args.save_model_dir, 'model_best_f1.pth.tar')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("==> loading model")
    num_classes = args.num_classes
    model = create_model(num_classes, args).to(device)
    model = load_checkpoint(model, checkpoint).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    input_data = args.input
    if os.path.isdir(input_data):
        input_folder_lst = [os.path.join(input_data, i) for i in os.listdir(input_data)]
        for input_folder in input_folder_lst:
            print(os.path.basename(input_folder))
            input_data = os.path.join(input_folder, 'patch.json')
            filter_patches(model, input_data, transform)
    else:
        filter_patches(model, input_data, transform)


def filter_patches(model, input_data, transform):
    dataset = DataLoader(input_data, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=64)
    tbar = tqdm(data_loader, desc='\r')

    model.eval()
    with torch.no_grad():
        probs = []
        preds = []
        paths = []

        for idx, (inputs, _, image_path) in enumerate(tbar):
            inputs = inputs.to(device)
            outputs = model(inputs).to(device)
            prob = F.softmax(outputs, dim=1)

            probs.extend(prob.tolist())
            preds.extend(torch.argmax(prob, dim=1).tolist())
            paths.extend(list(image_path))

        binary_preds = map(lambda x: 0 if x != 4 else 1, preds)

    result = [{'image_dir': img, 'pred': pred} for img, pred in zip(paths, binary_preds)]

    output_f = os.path.join(os.path.dirname(input_data), 'patch_pred.json')
    with open(output_f, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()




