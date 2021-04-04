from __future__ import print_function

import sys
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
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import models.resnet as resnet
import models.wideresnet as models
import models.mobileNetV2 as netv2
import models.senet as senet
from models.efficientnet import EfficientNet
import models.inceptionv4 as inceptionv4
from data_loader import DataLoader
from utils import AverageMeter, accuracy

from easydict import EasyDict as edict
from argparse import Namespace
import yaml

from utils import loss_func

config_dir = sys.argv[1]
config_file = os.path.basename(config_dir)
print('Test with ' + config_file)

with open(config_dir, 'r') as f:
    args = edict(yaml.load(f, Loader=yaml.FullLoader))

args = Namespace(**args)
args.expname = config_file.split('.yaml')[0]

output_csv_dir = os.path.join(args.output_csv_dir, args.expname)
if not os.path.exists(output_csv_dir):
    os.makedirs(output_csv_dir)

save_model_dir = os.path.join(output_csv_dir, 'models')
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)


def main():
    # Use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_set = DataLoader(args.test_list, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers)

    # Model
    print("==> creating model")
    num_classes = args.num_classes
    model = create_model(num_classes).to(device)


    criterion = nn.CrossEntropyLoss()
    criterion = loss_func.FocalLoss().cuda()

    # testing
    df = pd.DataFrame(columns=['exp', 'train', 'val', 'test', 'test_loss', 'test_acc', 'f1', 'mul_acc'])

    expname = args.expname.split('/')[-1]
    print('\n' + expname + ': TESTING!')
    train_set = os.path.basename(args.train_list).split('.')[0]
    val_set = os.path.basename(args.val_list).split('.')[0]
    test_set = os.path.basename(args.test_list).split('.')[0]

    model = load_checkpoint(model, save_model_dir, 'model_best_acc.pth.tar')
    test_loss, test_acc, f1, mul_acc = test(output_csv_dir, test_loader, model, criterion, device)

    df.loc[len(df)] = [expname, train_set, val_set, test_set, test_loss, test_acc, f1, mul_acc]
    output_csv_file = os.path.join(output_csv_dir, 'test.csv')
    df.to_csv(output_csv_file, index=False)


def create_model(num_classes):
    if args.network == 101:
        model = models.WideResNet(num_classes=num_classes)
    elif args.network == 102:
        model = resnet.resnet50()
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
        print('model not available! Using EfficientNet-b0 as default')
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    return model


def load_checkpoint(model, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def test(out_dir, test_loader, model, criterion, device):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    tbar = tqdm(test_loader, desc='\r')

    with torch.no_grad():
        correct = 0
        pred_history = []
        target_history = []
        name_history = []
        prob_history = []

        for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
            inputs = inputs.float()

            inputs, targets = inputs.to(device), targets.to(device)
            # compute output
            outputs = model(inputs)
            prob_out = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            [prec1, ] = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

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

            tbar.set_description('\r %s Loss: %.3f | Top1: %.3f' % ('Test Stats', losses.avg, top1.avg))

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