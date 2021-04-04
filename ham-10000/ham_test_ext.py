from __future__ import print_function

import argparse
import shutil
import time
import random
import json

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import pandas as pd


import models.resnet as resnet
import models.wideresnet as models
import models.mobileNetV2 as netv2
import models.senet as senet
from models.efficientnet import EfficientNet
import models.inceptionv4 as inceptionv4
from data_loader import DataLoader
import data_loader as dataset
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

from easydict import EasyDict as edict
from argparse import Namespace
import yaml



def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""

    with open(filename, 'r') as f:  # not valid grammar in Python 2.5
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return yaml_cfg


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')

parser.add_argument('--config', default='predict_new.yaml', help='config file')

args = parser.parse_args([])
config = 'Ext_MobileNet_os_v1_centermix_test.yaml'
print('Test with ' + config)

config_file = os.path.join('/Data/luy8/centermix/config_ham', config)
args = cfg_from_file(config_file)

args = Namespace(**args)
state = {k: v for k, v in args._get_kwargs()}
output_dir = args.output_csv_dir

pretrained_model_dir = ['/Data/luy8/centermix/exp_results/config_ham/MobileNet_os_v1_centermix_fold1',
                        '/Data/luy8/centermix/exp_results/config_ham/MobileNet_os_v1_centermix_fold2',
                        '/Data/luy8/centermix/exp_results/config_ham/MobileNet_os_v1_centermix_fold3',
                        '/Data/luy8/centermix/exp_results/config_ham/MobileNet_os_v1_centermix_fold4',
                        '/Data/luy8/centermix/exp_results/config_ham/MobileNet_os_v1_centermix_fold5']


def main():
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])

    val_set = DataLoader(args.test_list,
                         transform=transform_val,
                         split='val',
                         aug=args.aug,
                         aggregate=args.aggregate)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # Model
    print("==> creating model")

    num_classes = args.num_classes  # yaml

    def create_model(args, num_classes, pretrained_model):
        if args.network == 101:
            model = models.WideResNet(num_classes=num_classes)
        elif args.network == 102:
            model = resnet.resnet101(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif args.network == 103:
            model = netv2.mobilenet_v2(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif args.network == 104:
            model = senet.se_resnet50(num_classes=num_classes)
        elif args.network == 105:
            model = EfficientNet.from_pretrained(pretrained_model, num_classes=num_classes)
        elif args.network == 106:
            model = inceptionv4.inceptionv4(num_classes=num_classes, pretrained=None)

        model = model.cuda()

        return model

    # testing
    print('\nTESTING!')
    model_index = 1
    for pretrained_model in pretrained_model_dir:
        print('Test with ' + os.path.basename(pretrained_model))
        model = create_model(args, num_classes, pretrained_model)
        model = load_checkpoint(model, pretrained_model, 'model_best_acc.pth.tar')
        out_dir = os.path.join(output_dir, os.path.basename(pretrained_model).split('_')[-2])
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        validate(out_dir, val_loader, model, use_cuda, model_index)
        model_index += 1


def load_checkpoint(model, checkpoint, filename):
    filepath = os.path.join(checkpoint, 'models', filename)
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def validate(out_dir, valloader, model, use_cuda, model_index):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    tbar = tqdm(valloader, desc='\r')
    end = time.time()

    with torch.no_grad():
        correct = 0
        pred_history = []
        target_history = []
        name_history = []

        for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pred_clss = F.softmax(outputs, dim=1)
            pred = pred_clss.data.max(1)[1]  # ge
            correct += pred.eq(targets.data).cpu().sum()
            pred_history = np.concatenate((pred_history, pred.data.cpu().numpy()), axis=0)
            target_history = np.concatenate((target_history, targets.data.cpu().numpy()), axis=0)
            names = [os.path.basename(i).split('.')[0] for i in list(image_path)]
            name_history = np.concatenate((name_history, names), axis=0)

        df = pd.DataFrame(columns=['image', 'prediction'])
        df['image'] = name_history
        df['prediction'] = pred_history
        df.to_csv(os.path.join(out_dir, 'model' + str(model_index) + '.csv'), index=False)


if __name__ == '__main__':
    main()




