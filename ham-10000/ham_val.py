from __future__ import print_function

import sys
sys.path = ['/Data/luy8/centermix',
            '/snap/pycharm-professional/209/plugins/python/helpers/pycharm_display',
            '/home/hrlblab/anaconda3/envs/centermix/lib/python37.zip',
            '/home/hrlblab/anaconda3/envs/centermix/lib/python3.7',
            '/home/hrlblab/anaconda3/envs/centermix/lib/python3.7/lib-dynload',
            '/home/hrlblab/anaconda3/envs/centermix/lib/python3.7/site-packages',
            '/snap/pycharm-professional/209/plugins/python/helpers/pycharm_matplotlib_backend',
            '/usr/lib/python3.7'
            ]

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
from pandas import DataFrame
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import models.resnet as resnet
import models.wideresnet as models
import models.mobileNetV2 as netv2
import models.senet as senet
from models.efficientnet import EfficientNet
import models.inceptionv4 as inceptionv4
from data_loader import DataLoader
import data_loader as dataset
from utils import AverageMeter, accuracy, mkdir_p

from easydict import EasyDict as edict
from argparse import Namespace
import yaml

from utils import focal_loss
from PIL import Image


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""

    with open(filename, 'r') as f:  # not valid grammar in Python 2.5
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return yaml_cfg


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')

parser.add_argument('--config', default='EfficientNetb1_none_none_fold1.yaml', help='config file')

args = parser.parse_args([])
config = sys.argv[1]
print('Test with ' + config)

config_file = os.path.join('/Data/luy8/centermix/config_ham/', config)
args = cfg_from_file(config_file)

args = Namespace(**args)
state = {k: v for k, v in args._get_kwargs()}
args.expname = config.replace('.yaml', '')
output_dir = os.path.join(args.output_csv_dir, args.expname)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_model_dir = os.path.join(output_dir, 'models')
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)


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

    def create_model(args, num_classes):
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
            model = EfficientNet.from_pretrained(sys.argv[2], num_classes=num_classes)
        elif args.network == 106:
            model = inceptionv4.inceptionv4(num_classes=num_classes, pretrained=None)
        else:
            raise ValueError('No corresponding model')

        model = model.cuda()

        return model

    model = create_model(args, num_classes=num_classes)
    criterion = focal_loss.FocalLoss().cuda()

    # testing
    df = pd.DataFrame(columns=['exp', 'train', 'val', 'test', 'test_loss', 'test_acc', 'f1', 'mul_acc'])

    expname = args.expname.split('/')[-1]
    print('\n' + expname + ': TESTING!')
    train_set = os.path.basename(args.train_list).split('.')[0]
    val_set = os.path.basename(args.val_list).split('.')[0]
    test_set = os.path.basename(args.test_list).split('.')[0]

    best_bal_mul_acc = 0
    model_dir = os.listdir(save_model_dir)
    model_dir.sort()
    for saved_model in model_dir:
        print("Loading {}...".format(saved_model))
        model = load_checkpoint(model, save_model_dir, saved_model)
        test_loss, test_acc, f1, mul_acc = validate(output_dir,
                                                  val_loader,
                                                  model,
                                                  criterion,
                                                  use_cuda,
                                                  mode='Test Stats')

        df.loc[len(df)] = [saved_model, train_set, val_set, test_set, test_loss, test_acc, f1, mul_acc]
        if mul_acc > best_bal_mul_acc:
            best_bal_mul_acc = mul_acc
            shutil.copyfile(os.path.join(save_model_dir, saved_model), os.path.join(save_model_dir, 'model_best_acc.pth.tar'))

    output_csv_file = os.path.join(output_dir, 'full_test.csv')
    df.to_csv(output_csv_file, index=False)


def load_checkpoint(model, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def validate(out_dir, valloader, model, criterion, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    tbar = tqdm(valloader, desc='\r')
    end = time.time()

    with torch.no_grad():
        correct = 0
        pred_history = []
        target_history = []
        name_history = []
        prob_history = []

        for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.float()

            if use_cuda:
                 inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            prob_out = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            [prec1, ] = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pred_clss = F.softmax(outputs, dim=1)
            pred = pred_clss.data.max(1)[1]  # ge
            correct += pred.eq(targets.data).cpu().sum()
            pred_history = np.concatenate((pred_history, pred.data.cpu().numpy()), axis=0)
            target_history = np.concatenate((target_history, targets.data.cpu().numpy()), axis=0)
            name_history = np.concatenate((name_history, image_path), axis=0)
            if batch_idx == 0:
                prob_history = prob_out.data.cpu().numpy()
            else:
                prob_history = np.concatenate((prob_history, prob_out.data.cpu().numpy()), axis=0)

            tbar.set_description('\r %s Loss: %.3f | Top1: %.3f' % (mode, losses.avg, top1.avg))

        f1s = f1_score(target_history, pred_history, average=None)
        f1_avg = sum(f1s) / len(f1s)
        mul_acc = balanced_accuracy_score(target_history, pred_history)
        epoch_summary(out_dir, name_history, pred_history, target_history)

    return losses.avg, top1.avg, f1_avg, mul_acc


# output csv file for result in each epoch
def epoch_summary(out_dir, name_history, pred_history, target_history):
    csv_file_name = os.path.join(out_dir, 'epoch_test.csv')

    df = pd.DataFrame()
    df['image'] = name_history
    df['prediction'] = pred_history
    df['target'] = target_history
    df.to_csv(csv_file_name)


if __name__ == '__main__':
    main()




