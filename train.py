import os
import sys
import time
import shutil
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F

import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import compute_class_weight
import models.resnet as resnet
import models.resnetv2 as resnetv2
import models.mobilenetv2 as mobilenetv2
import microsoftvision
from models.efficientnet import EfficientNet
from data_loader import DataLoader
from utils.augmentation import GaussianBlur, rand_bbox, coordinate, polygon_vertices
from utils import AverageMeter, accuracy
import argparse
from easydict import EasyDict as edict
from argparse import Namespace
import yaml
from utils import loss
from utils.optimizer import SAM
from utils.torchsampler.imbalanced import ImbalancedDatasetSampler
from torch.utils.tensorboard import SummaryWriter

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='configuration file')
parser.add_argument('--bit_model', default=None, help='BiT model')

cmd_args = parser.parse_args()
config_dir = cmd_args.config
config_file = os.path.basename(config_dir)
print('Train with ' + config_file)

with open(config_dir, 'r') as f:
    args = edict(yaml.load(f, Loader=yaml.FullLoader))

args.expname = config_file.split('.yaml')[0]
args.hyperparam_tune = True
args.optimizer = 'ADAM'
args.resample = True
args.loss = 'Focal'
args.weighted_loss = False

output_csv_dir = os.path.join(args.output_csv_dir, args.expname)
os.makedirs(output_csv_dir, exist_ok=True)

log_dir = os.path.join(output_csv_dir, 'log')
os.makedirs(log_dir, exist_ok=True)

save_model_dir = os.path.join(output_csv_dir, 'models')
os.makedirs(save_model_dir, exist_ok=True)


def main(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(10)

    if args.hyperparam_tune:
        batch_size = config['batch_size']
        lr = config['lr']
    else:
        batch_size = args.batch_size
        lr = args.lr

    best_f1 = 0

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        GaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    train_set = DataLoader(args.train_list, transform=transform_train)

    if args.resample:
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   sampler=ImbalancedDatasetSampler(train_set,
                                                                                    num_samples=len(train_set),
                                                                                    callback_get_label=train_set.data),
                                                   num_workers=args.num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers)

    val_set = DataLoader(args.val_list, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # Load model
    print("==> Creating model")
    num_classes = args.num_classes
    model = create_model(num_classes).to(device)

    # choose loss function
    if args.weighted_loss:
        targets = [i['target'] for i in train_set.data]
        weights = compute_class_weight('balanced', classes=np.unique(targets), y=np.array(targets))
        criterion = select_loss_func(choice=args.loss, weights=torch.tensor(weights, dtype=torch.float))
    else:
        criterion = select_loss_func(choice=args.loss)

    # choose optimizer
    print('==> {} optimizer'.format(args.optimizer))
    if args.optimizer == 'SAM':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9)
    elif args.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

    # set up logger
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = args.start_epoch
    if args.dataset == 'renal':
        df = pd.DataFrame(columns=['model', 'lr', 'epoch_num', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                                   'normal', 'obsolescent', 'solidified', 'disappearing', 'non_glom', 'f1'])

    elif args.dataset == 'ham':
        df = pd.DataFrame(columns=['model', 'lr', 'epoch_num', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                                   'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'mul_acc', 'f1'])
    else:
        raise ValueError('no such dataset exists!')

    # start training
    for epoch in range(start_epoch, args.epochs):
        epoch += 1

        if args.optimizer != 'ADAM':
            cur_lr = adjust_learning_rate(lr, optimizer, epoch)
        else:
            cur_lr = lr

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, cur_lr))

        train_loss, train_acc, train_f1, train_f1s = train(train_loader, model, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_f1s = validate(output_csv_dir, val_loader, model, criterion, epoch, device)

        if args.hyperparam_tune:
            tune.report(loss=val_loss, accuracy=val_f1)

        writer.add_scalars("loss/", {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars("f1/", {'train': train_f1, 'val': val_f1}, epoch)

        # write to csv
        df.loc[epoch] = [args.network, cur_lr, epoch, train_loss, val_loss, train_acc, val_acc] + val_f1s + [val_f1]

        output_csv_file = os.path.join(output_csv_dir, 'output.csv')
        df.to_csv(output_csv_file, index=False)

        # save model
        is_best_f1 = val_f1 > best_f1
        best_f1 = max(val_f1, best_f1)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_f1': best_f1,
            'optimizer': optimizer.state_dict(),
        }, is_best_f1, epoch, save_model_dir)

    print('Best acc:')
    print(best_acc)


def create_model(num_classes):
    if args.network == 100:
        model = resnet.resnet18(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    if args.network == 101:
        model = resnet.resnet50(pretrained=args.pretrain)
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
        print('model not available! Using PyTorch ResNet50 as default')
        model = resnet.resnet50(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def select_loss_func(choice='CrossEntropy', weights=None):
    if weights is not None:
        print("==> Weighted {} loss".format(choice))
    else:
        print("==> {} loss".format(choice))

    if choice == 'Focal':
        return loss.FocalLoss(alpha=weights, gamma=2, reduce=True).cuda()
    elif choice == 'Class-Balanced':
        return loss.EffectiveSamplesLoss(beta=0.999,
                                         num_cls=args.num_classes,
                                         sample_per_cls=np.array([500, 300, 20, 30, 400]),
                                         focal=False,
                                         focal_gamma=2,
                                         focal_alpha=4).cuda()
    else:
        return nn.CrossEntropyLoss(weight=weights).cuda()


def train(train_loader, model, optimizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    top1 = AverageMeter()
    tbar = tqdm(train_loader, desc='\r')

    model.train()
    preds = []
    gts = []
    paths = []
    for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        r = np.random.rand(1)
        if args.circlemix_prob > r:
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

            assert height == width, 'height does not equal to width'
            side = height

            mask = np.zeros((side, side), np.uint8)
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

            if args.optimizer == 'SAM':
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                (criterion(model(inputs), target_a) * (1. - lam) + criterion(model(inputs), target_b) * lam).backward()
                optimizer.second_step(zero_grad=True)
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        elif args.cutmix_prob > r:
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

            if args.optimizer == 'SAM':
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                (criterion(model(inputs), target_a) * (1. - lam) + criterion(model(inputs), target_b) * lam).backward()
                optimizer.second_step(zero_grad=True)
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        elif args.cutout_prob > r:
            lam = np.random.beta(args.beta, args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = 0

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if args.optimizer == 'SAM':
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                criterion(model(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        else:
            inputs = inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if args.optimizer == 'SAM':
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                criterion(model(inputs), targets).backward()  # make sure to do a full forward pass
                optimizer.second_step(zero_grad=True)
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        score = F.softmax(outputs, dim=1)
        pred = score.data.max(1)[1]

        preds.extend(pred.tolist())
        gts.extend(targets.tolist())
        paths.extend(image_path)

        # record loss
        [acc1, ] = accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        tbar.set_description('\r Train Loss: %.3f | Top1: %.3f' % (losses.avg, top1.avg))

    f1s = list(f1_score(gts, preds, average=None))
    f1_avg = sum(f1s) / len(f1s)

    return losses.avg, top1.avg, f1_avg, f1s


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
        preds = []
        gts = []
        paths = []

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

            score = F.softmax(outputs, dim=1)
            pred = score.data.max(1)[1]
            preds.extend(pred.tolist())
            gts.extend(targets.tolist())
            paths.extend(image_path)

            tbar.set_description('\r %s Loss: %.3f | Top1: %.3f' % ('Valid Stats', losses.avg, top1.avg))

        f1s = list(f1_score(gts, preds, average=None))
        f1_avg = sum(f1s)/len(f1s)

        epoch_summary(out_dir, epoch, paths, preds, gts)

    return losses.avg, top1.avg, f1_avg, f1s


def adjust_learning_rate(lr, optimizer, epoch):
    cur_lr = lr * (0.1 ** ( (epoch - 1) // 30 ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


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


def save_checkpoint(state, is_best_f1, epoch, checkpoint, filename='checkpoint.pth.tar'):
    filename = 'epoch' + str(epoch) + '_' + filename
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best_f1:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_f1.pth.tar'))


def load_checkpoint(model, filepath):
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    if args.hyperparam_tune:
        max_num_epochs = 10
        gpus_per_trial = 1

        config = {
            "lr": tune.grid_search([1e-4, 1e-3, 1e-2]),
            "batch_size": tune.grid_search([8, 16])
        }
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)

        result = tune.run(
            tune.with_parameters(main),
            resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
            config=config,
            metric="loss",
            mode="min",
            scheduler=scheduler,
            local_dir=output_csv_dir
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    else:
        config = {}
        main(config)