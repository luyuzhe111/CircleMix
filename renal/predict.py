from __future__ import print_function
import sys
import time
import random
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
import models.wideresnet as models
import models.mobileNetV2 as netv2
import models.senet as senet
from models.efficientnet import EfficientNet
import models.inceptionv4 as inceptionv4
from data_loader import DataLoader
from utils import AverageMeter, accuracy, mkdir_p

from easydict import EasyDict as edict
from argparse import Namespace
import yaml


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""

    with open(filename, 'r') as f:  # not valid grammar in Python 2.5
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return yaml_cfg


config_dir = sys.argv[1]
config = os.path.basename(config_dir)
print('Test with ' + config)

args = cfg_from_file(config_dir)
args = Namespace(**args)

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

    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_set = DataLoader(args.test_list, transform=transform_val)
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

    expname = args.expname.split('/')[-1]
    print('\n' + expname + ': Predicting!')

    model = load_checkpoint(model, save_model_dir, 'model_best_acc.pth.tar')
    validate(output_dir, val_loader, model, use_cuda, mode='Test Stats')


def load_checkpoint(model, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def validate(out_dir, valloader, model, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    tbar = tqdm(valloader, desc='\r')
    end = time.time()

    with torch.no_grad():
        pred_history = []
        target_history = []
        name_history = []

        for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.float()

            if use_cuda:
                 inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pred_clss = F.softmax(outputs, dim=1)
            pred = pred_clss.data.max(1)[1]  # ge
            pred_history = np.concatenate((pred_history, pred.data.cpu().numpy()), axis=0)
            target_history = np.concatenate((target_history, targets.data.cpu().numpy()), axis=0)
            name_history = np.concatenate((name_history, image_path), axis=0)

            tbar.set_description('\r %s Loss: %.3f | Top1: %.3f' % (mode, losses.avg, top1.avg))

        epoch_summary(out_dir, name_history, pred_history, target_history)


# output csv file for result in each epoch
def epoch_summary(out_dir, name_history, pred_history, target_history):
    csv_file_name = os.path.join(out_dir, 'prediction.csv')

    df = pd.DataFrame()
    df['image'] = name_history
    df['target'] = target_history
    df['prediction'] = pred_history
    if args.num_classes < 5:
        hierarchy = config.split('_')[0].split('-')[1]
        if hierarchy == 'NC2':
            df['prediction'] = df['prediction'] * 4
        if hierarchy == 'C3':
            df['prediction'] = df['prediction'] + 1
        if hierarchy == 'C2':
            df['prediction'] = df['prediction'] + 2

    df.to_csv(csv_file_name)


if __name__ == '__main__':
    main()




