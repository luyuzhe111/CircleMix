from __future__ import print_function

import sys
import shutil
import time

import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
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
print('Train with ' + config_file)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_acc = 0
    best_f1 = 0

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_set = DataLoader(args.train_list, transform=transform_train, split='train')
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)

    val_set = DataLoader(args.val_list, transform=transform_val, split='val')
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # Load model
    print("==> Creating model")
    print('==> {} optimizer'.format(args.optimizer))
    criterion = select_loss_func()

    num_classes = args.num_classes
    model = create_model(num_classes).to(device)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    start_epoch = args.start_epoch

    # output performance of a model for each epoch
    df = pd.DataFrame(columns=['model', 'lr', 'epoch_num', 'train_loss',
                               'val_loss', 'train_acc', 'val_acc', 'f1', 'mul_acc'])

    for epoch in range(start_epoch, args.epochs):
        epoch += 1

        cur_lr = adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, cur_lr))

        train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
        val_loss, val_acc, f1, mul_acc = validate(output_csv_dir, val_loader, model, criterion, epoch, device)

        # write to csv
        df.loc[epoch] = [args.network, cur_lr, epoch, train_loss, val_loss, train_acc, val_acc, f1, mul_acc]

        output_csv_file = os.path.join(output_csv_dir, 'output.csv')
        df.to_csv(output_csv_file, index=False)

        # save model
        is_best_f1 = f1 > best_f1
        best_f1 = max(f1, best_f1)
        is_best_acc = mul_acc > best_acc
        best_acc = max(mul_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best_acc, is_best_f1, epoch, save_model_dir)

    print('Best acc:')
    print(best_acc)


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


def select_loss_func(choice='CrossEntropy'):
    print("==> {} loss".format(choice))
    if choice == 'Focal':
        return loss_func.FocalLoss(alpha=1, gamma=2, reduce=True).cuda()
    elif choice == 'Class-Balanced':
        return loss_func.EffectiveSamplesLoss(beta=0.999,
                                              num_cls=args.num_classes,
                                              sample_per_cls=np.array([500, 300, 20, 30, 400]),
                                              focal=False,
                                              focal_gamma=2,
                                              focal_alpha=4).cuda()
    else:
        return nn.CrossEntropyLoss().cuda()


def train(train_loader, model, optimizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    top1 = AverageMeter()
    tbar = tqdm(train_loader, desc='\r')

    model.train()
    for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        r = np.random.rand(1)
        if args.centermix_prob > r:
            # generate circlemix sample
            rand_index = torch.randperm(inputs.size()[0])
            target_a = targets
            target_b = targets[rand_index]

            r1 = np.random.randint(0, 360)
            r2 = np.random.randint(0, 360)
            start, end = min(r1, r2), max(r1, r2)
            lam = (end - start) / 360

            height = inputs.shape[2]
            width = inputs.shape[3]

            if inputs.dtype == torch.float32:
                mask = np.zeros((height, width), np.float32)
            else:
                mask = np.zeros((height, width), np.uint8)

            assert height == width, 'height does not equal to width'
            side = height

            vertices = polygon_vertices(side, start, end)

            roi_mask = cv2.fillPoly(mask, np.array([vertices]), 255)
            roi_mask_rgb = np.repeat(roi_mask[np.newaxis, :, :], inputs.shape[1], axis=0)
            roi_mask_batch = np.repeat(roi_mask_rgb[np.newaxis, :, :, :], inputs.shape[0], axis=0)
            roi_mask_batch = torch.from_numpy(roi_mask_batch)

            roi_mask_batch = roi_mask_batch.to(device)
            rand_index = rand_index.to(device)

            inputs2 = inputs[rand_index].clone()
            inputs[roi_mask_batch > 0] = inputs2[roi_mask_batch > 0]

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * (1. - lam) + criterion(outputs, target_b) * lam

        # compute output
        elif args.beta > 0 and args.cutmix_prob > r:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)

        else:
            inputs = inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        [acc1, ] = accuracy(outputs, targets, topk=(1,))
        # print('acc1 = %.2f' % acc1)
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        tbar.set_description('\r Train Loss: %.3f | Top1: %.3f' % (losses.avg, top1.avg))

    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 100 at every 1/3 of all epochs"""
    lr = args.lr * (0.1 ** ( (epoch - 1) // (args.epochs * 1/3)) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def polygon_vertices(size, start, end):
    side = size-1
    center = (int(side/2), int(side/2))
    bound1 = coordinate(start, side)
    bound2 = coordinate(end, side)

    square_vertices = {90: (0, side), 180: (side, side), 270: (side, 0)}
    inner_vertices = []
    if start < 90 < end:
        inner_vertices.append(square_vertices[90])
    if start < 180 < end:
        inner_vertices.append(square_vertices[180])
    if start < 270 < end:
        inner_vertices.append(square_vertices[270])

    all_vertices = [center] + [bound1] + inner_vertices + [bound2]
    return all_vertices


def coordinate(num, side):
    length = side + 1
    if 0 <= num < 90:
        return 0, int(num/90*length)
    elif 90 <= num < 180:
        return int((num-90)/90*length), side
    elif 180 <= num < 270:
        return side, side - int((num-180)/90*length)
    elif 270 <= num <= 360:
        return side - int((num - 270)/90*length), 0


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(out_dir, val_loader, model, criterion, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    tbar = tqdm(val_loader, desc='\r')
    end = time.time()

    with torch.no_grad():
        pred_history = []
        target_history = []
        name_history = []

        for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.float()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            [prec1,] = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pred_clss = F.softmax(outputs, dim=1)
            pred = pred_clss.data.max(1)[1]  # ge
            pred_history = np.concatenate((pred_history, pred.data.cpu().numpy()), axis=0)
            target_history = np.concatenate((target_history, targets.data.cpu().numpy()), axis=0)
            name_history = np.concatenate((name_history, image_path), axis=0)

            tbar.set_description('\r %s Loss: %.3f | Top1: %.3f' % ('Valid Stats', losses.avg, top1.avg))

        f1s = f1_score(target_history, pred_history, average=None)
        f1_avg = sum(f1s)/len(f1s)

        mul_acc = balanced_accuracy_score(target_history, pred_history)

        epoch_summary(out_dir, epoch, name_history, pred_history, target_history)

    return losses.avg, top1.avg, f1_avg, mul_acc


# output csv file for result in each epoch
def epoch_summary(out_dir, epoch, name_history, pred_history, target_history):
    epoch_dir = os.path.join(out_dir, 'epochs')
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    csv_file_name = os.path.join(epoch_dir, 'epoch_%04d.csv' % epoch)

    df = pd.DataFrame()
    df['image'] = name_history
    df['prediction'] = pred_history
    df['target'] = target_history
    df.to_csv(csv_file_name)


def save_checkpoint(state, is_best_acc, is_best_f1, epoch, checkpoint, filename='checkpoint.pth.tar'):
    filename = 'epoch' + str(epoch) + '_' + filename
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best_acc:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_acc.pth.tar'))
    if is_best_f1:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_f1.pth.tar'))


def load_checkpoint(model, filepath):
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    main()