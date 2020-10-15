from __future__ import print_function

import math
import sys
sys.path.append('/home/cheny82/anaconda3/envs/torch11/lib/python3.7/site-packages')

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
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import f1_score

import models.resnet as resnet
import models.wideresnet as models
import models.mobileNetV2 as netv2
import models.senet as senet
from models.efficientnet import EfficientNet
import models.inceptionv4 as inceptionv4
from data_loader import DataLoader
import data_loader as dataset
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

from easydict import EasyDict as edict
from argparse import Namespace
import yaml

from utils import focal_loss

import imgaug.augmenters as iaa
from data_loader import normalise
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
# config = 'MobileNet_none_none_fold1.yaml'
print('Train with ' + config)
config_file = os.path.join('/Data/luy8/centermix/config/', config)
args = cfg_from_file(config_file)

args = Namespace(**args)
state = {k: v for k, v in args._get_kwargs()}
# config = os.path.join('config', args.config)
args.expname = config.replace('.yaml', '')

output_csv_dir = os.path.join(args.output_csv_dir, 'config', args.expname)
if not os.path.exists(output_csv_dir):
    os.makedirs(output_csv_dir)

save_model_dir = os.path.join(output_csv_dir, 'models')
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)


def main():
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()

    # best test accuracy
    # global best_acc
    best_acc = 0
    best_f1 = 0

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing kidpath')
    transform_train = transforms.Compose([
        # dataset.Resize((size, size)),
        # dataset.RandomPadandCrop(size),
        # dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

    transform_val = transforms.Compose([
        # dataset.Resize((size, size)),
        dataset.ToTensor(),
    ])

    train_labeled_set = DataLoader(args.train_list, transform=transform_train, split='train')

    labeled_trainloader = torch.utils.data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.num_workers)

    val_set = DataLoader(args.val_list, transform=transform_val, split='val')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    # Model
    print("==> creating model")

    num_classes = args.num_classes
    def create_model(args, num_classes, ema=False):
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
            # model = EfficientNet.from_name('efficientnet-b0')
            model = EfficientNet.from_pretrained(sys.argv[2], num_classes=num_classes)
        elif args.network == 106:
            model = inceptionv4.inceptionv4(num_classes=num_classes, pretrained=None)

        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(args, num_classes=num_classes)
    ema_model = create_model(args, num_classes=num_classes, ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = focal_loss.FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    # Resume
    title = 'noisy-kidpath'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
        logger.set_names(['Train Loss', 'Valid Loss', 'Valid Acc'])
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Valid Loss', 'Valid Acc'])

    writer = SummaryWriter(args.out)
    test_accs = []

    # output performance of a model with each epoch

    # Train and val
    df = pd.DataFrame(columns=['cur_model', 'supervised', 'epoch_num', 'train_loss',
                               'val_loss', 'val_acc', 'train_acc', 'f1_0', 'f1_1', 'f1_avg'])

    for epoch in range(start_epoch, args.epochs):
        epoch += 1

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))

        train_loss, train_acc = train_supervise(labeled_trainloader, model, optimizer, criterion, use_cuda)
        val_loss, val_acc, f1s, f1_avg = validate(output_csv_dir, val_loader, model, criterion, epoch, use_cuda, mode='Valid Stats')

        step = args.val_iteration * (epoch)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        # writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)

        logger.append([train_loss, val_loss, val_acc])

        # write to csv
        f1_0, f1_1 = f1s
        df.loc[epoch] = [args.network, args.supervised, epoch, train_loss,
                         val_loss, val_acc, train_acc, f1_0, f1_1, f1_avg]

        output_csv_file = os.path.join(output_csv_dir, 'output.csv')
        df.to_csv(output_csv_file, index=False)

        # save model
        is_best = f1_avg > best_f1
        best_f1 = max(f1_avg, best_f1)
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, save_model_dir)
        test_accs.append(val_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def train_supervise(labeled_trainloader, model, optimizer, criterion, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    top1 = AverageMeter()
    tbar = tqdm(labeled_trainloader, desc='\r')

    model.train()
    for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
        # measure data loading time
        data_time.update(time.time() - end)
        #
        inputs = inputs.float()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        r = np.random.rand(1)
        if args.centermix_prob > r:
            # generate circlemix sample
            rand_index = torch.randperm(inputs.size()[0])
            target_a = targets
            target_b = targets[rand_index]

            r1 = np.random.randint(0, 360)
            r2 = np.random.randint(0, 360)
            start = min(r1, r2)
            end = max(r1, r2)
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
            roi_mask_batch = torch.tensor(roi_mask_batch)

            if use_cuda:
                roi_mask_batch = roi_mask_batch.cuda()
                rand_index = rand_index.cuda()

            inputs2 = inputs[rand_index].clone()
            inputs[roi_mask_batch > 0] = inputs2[roi_mask_batch > 0]

            # # common data augmentation
            # seq = iaa.Sequential([
            #     # iaa.ChannelShuffle(0.35),
            #     iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
            #     iaa.Affine(rotate=(-180, 180)),
            #     # iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            #     # iaa.Affine(shear=(-16, 16)),
            #     iaa.Fliplr(0.5),
            #     iaa.GaussianBlur(sigma=(0, 1.0))
            # ])
            #
            # transform_train = transforms.Compose([
            #     dataset.ToTensor(),
            # ])
            #
            # inputs = inputs.cpu().numpy()
            # inputs_normed = np.zeros(inputs.shape)
            # for i in range(inputs.shape[0]):
            #     tmp = inputs[i]
            #     tmp = np.transpose(tmp, (1, 2, 0))
            #     tmp_img = np.expand_dims(tmp, axis=0)
            #     tmp_img = seq(images=tmp_img)
            #
            #     # if we would like to see the data augmentation
            #     # seq.show_grid([tmp_img[0]], cols=16, rows=8)
            #
            #     tmp = np.transpose(normalise(tmp_img[0]), (2, 0, 1))
            #     # tmp = transform_train(tmp)
            #     inputs_normed[i] = tmp
            #
            # # compute output
            # inputs = transform_train(inputs_normed)
            inputs = inputs.float()
            inputs = inputs.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * (1. - lam) + criterion(outputs, target_b) * lam

        # compute output
        elif args.beta > 0 and args.cutmix_prob > r:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

            seq = iaa.Sequential([
                # iaa.ChannelShuffle(0.35),
                iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                iaa.Affine(rotate=(-180, 180)),
                # iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                # iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.GaussianBlur(sigma=(0, 1.0))
            ])

            transform_train = transforms.Compose([
                dataset.ToTensor(),
            ])

            # inputs_normed = np.zeros(inputs.shape)
            # inputs = inputs.float()
            inputs = inputs.cpu().numpy()
            inputs_normed = np.zeros(inputs.shape)
            for i in range(inputs.shape[0]):
                tmp = inputs[i]
                tmp = np.transpose(tmp, (1, 2, 0))
                tmp_img = np.expand_dims(tmp, axis=0)
                tmp_img = seq(images=tmp_img)

                # if we would like to see the data augmentation
                # seq.show_grid([tmp_img[0]], cols=16, rows=8)

                tmp = np.transpose(normalise(tmp_img[0]), (2, 0, 1))
                # tmp = transform_train(tmp)
                inputs_normed[i] = tmp

            # compute output
            inputs = transform_train(inputs_normed)
            inputs = inputs.float()
            inputs = inputs.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)

        else:
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


def validate(out_dir, valloader, model, criterion, epoch, use_cuda, mode):
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
            [prec1,] = accuracy(outputs, targets, topk=(1,))
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

        f1s = f1_score(pred_history, target_history, average=None)
        f1_avg = sum(f1s)/len(f1s)
        epoch_summary(out_dir, epoch, name_history, pred_history, target_history, prob_history, top1.avg)

    return losses.avg, top1.avg, f1s, f1_avg


# output csv file for result in each epoch
def epoch_summary(out_dir, epoch, name_history, pred_history, target_history, prob_history, acc):
    epoch_dir = os.path.join(out_dir, 'epochs')
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    csv_file_name = os.path.join(epoch_dir, 'epoch_%04d.csv' % epoch)

    df = pd.DataFrame()
    df['image'] = name_history
    df['prediction'] = pred_history
    df['target'] = target_history

    # columns = ['image', 'prediction', 'true', 'acc']
    # for pi in range(prob_history.shape[1]):
    #     columns = columns + ['clss_%d'%(pi)]
    #
    # df = DataFrame(columns=columns)
    # for i in range(len(pred_history)):
    #     # df.loc[i] = [name_history[i], pred_history[i], target_history[i]]
    #     df.loc[i] = [name_history[i], pred_history[i], target_history[i], acc] + list(prob_history[i])
    df.to_csv(csv_file_name)


def save_checkpoint(state, is_best, epoch, checkpoint, filename='checkpoint.pth.tar'):
    filename = 'epoch' + str(epoch) + '_' + filename
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def load_checkpoint(model, filepath):
    # filename = 'epoch' + str(epoch) + '_' + filename
    # filepath = os.path.join(checkpoint, filename)
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    main()




