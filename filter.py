import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import json

import models.resnet as resnet
import models.resnetv2 as resnetv2
import microsoftvision
from data_loader import DataLoader

from easydict import EasyDict as edict
import argparse
from argparse import Namespace
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='list of patches to be filtered')
parser.add_argument('--config', required=True, help='configuration file')
parser.add_argument('--setting', required=True, help='experiment setting')
parser.add_argument('--bit_model', default=None, help='BiT model')

cmd_args = parser.parse_args()
config_dir = cmd_args.config
config_file = os.path.basename(config_dir)
print('Train with ', cmd_args.setting, config_file)

with open(config_dir, 'r') as f:
    args = edict(yaml.load(f, Loader=yaml.FullLoader))

args = Namespace(**args)
args.expname = config_file.split('.yaml')[0]

output_csv_dir = os.path.join(args.output_csv_dir, 'without_fibrosis', cmd_args.setting, args.expname)
if not os.path.exists(output_csv_dir):
    os.mkdir(output_csv_dir)

save_model_dir = os.path.join(output_csv_dir, 'models')
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

pretrained_model = os.path.join(save_model_dir, 'model_best_f1.pth.tar')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("==> loading model")
    num_classes = args.num_classes
    model = create_model(num_classes).to(device)
    model = load_checkpoint(model, pretrained_model).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    input_data = cmd_args.input

    if os.path.isdir(input_data):
        input_folder_lst = [os.path.join(input_data, i) for i in os.listdir(input_data)]
        for input_folder in input_folder_lst:
            print(os.path.basename(input_folder))
            input_data = os.path.join(input_folder, 'patch.json')
            filter_patches(model, input_data, transform)
    else:
        filter_patches(model, input_data, transform)


def create_model(num_classes):
    if args.network == 101:
        model = resnet.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif args.network == 102:
        architecture = os.path.basename(cmd_args.bit_model)
        model = resnetv2.KNOWN_MODELS[architecture.split('.')[0]](head_size=num_classes, zero_head=True)
        model.load_from(np.load(cmd_args.bit_model))
        print(f'Load pre-trained model {cmd_args.bit_model}')
    elif args.network == 103:
        model = microsoftvision.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    else:
        print('model not available! Using ResNet-50 as default')
        model = resnet.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def load_checkpoint(model, filepath):
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def filter_patches(model, input_data, transform):
    dataset = DataLoader(input_data, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)
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




