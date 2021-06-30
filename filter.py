import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import pandas as pd

import models.resnet as resnet
import models.resnetv2 as resnetv2
import microsoftvision
from data_loader import DataLoader
from utils import AverageMeter

from easydict import EasyDict as edict
import argparse
from argparse import Namespace
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--det_result_dir', required=True, help='detection result dir')
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
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    det_result_dir = args.det_result_dir
    for folder_by_scn in os.listdir(det_result_dir):
        folder_by_scn_dir = os.path.join(det_result_dir, folder_by_scn)
        filter_det_result(model, folder_by_scn_dir, transform)


    # test_set = DataLoader(args.ext_test_list, transform=transform)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    #
    # print('\nTest with ' + pretrained_model.split('/')[-3])
    #
    #
    #
    # predict(output_csv_dir, test_loader, model, device)


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


def filter_det_result(model, folder_by_scn_dir, transform):
    patch_lst = [{'image_dir': f'{folder_by_scn_dir}/{i}', 'target': -1} for i in os.listdir(folder_by_scn_dir)]

    dataset = DataLoader(patch_lst, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(patch_lst), shuffle=False)

    model.eval()
    with torch.no_grad():
        pred_history = []
        target_history = []
        name_history = []
        prob_history = []

        for (inputs, _, image_path) in enumerate(data_loader):

            inputs = inputs.to(device)
            outputs = model(inputs)

            prob = F.softmax(outputs, dim=1)
            pred = prob.data.max(1)[1]
            pred_history = np.concatenate((pred_history, pred.data.cpu().numpy()), axis=0)
            target_history = np.concatenate((target_history, targets.data.cpu().numpy()), axis=0)
            names = [os.path.basename(i).split('.')[0] for i in list(image_path)]
            name_history = np.concatenate((name_history, names), axis=0)

            if batch_idx == 0:
                prob_history = prob.data.cpu().numpy()
            else:
                prob_history = np.concatenate((prob_history, prob.data.cpu().numpy()), axis=0)

        if args.dataset == 'renal':
            prob_pos = prob_history[:, 1] + prob_history[:, 2] + prob_history[:, 3]
            data = np.concatenate((name_history[..., np.newaxis],
                                   prob_history,
                                   prob_pos[..., np.newaxis],
                                   target_history[..., np.newaxis]), axis=1)
            if args.num_classes == 4:
                df = pd.DataFrame(data, columns=['image',
                                                 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'pro_pos', 'target'])
            elif args.num_classes == 5:
                df = pd.DataFrame(data, columns=['image',
                                                 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4', 'pro_pos', 'target'])

        else:
            data = np.concatenate((name_history[..., np.newaxis],
                                   pred_history[..., np.newaxis],
                                   target_history[..., np.newaxis]), axis=1)
            df = pd.DataFrame(data, columns=['image', 'prediction', 'target'])

        df.to_csv(os.path.join(out_dir, 'predict_f1.csv'), index=False)


if __name__ == '__main__':
    main()




