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
from sklearn.metrics import confusion_matrix
import models.resnet as resnet
import models.resnetv2 as resnetv2
import microsoftvision
import models.mobilenetv2 as mobilenetv2
from models.efficientnet import EfficientNet
import models.inceptionv4 as inceptionv4
from data_loader import DataLoader
from utils import AverageMeter, accuracy
import argparse
from easydict import EasyDict as edict
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='configuration file')
parser.add_argument('--bit_model', default=None, help='BiT model')
parser.add_argument('--average', default=False, type=bool, help='whether to average top3 models')

cmd_args = parser.parse_args()
config_dir = cmd_args.config
config_file = os.path.basename(config_dir)
print('Train with ' + config_file)

with open(config_dir, 'r') as f:
    args = edict(yaml.load(f, Loader=yaml.FullLoader))

args.expname = config_file.split('.yaml')[0]

output_csv_dir = os.path.join(args.output_csv_dir, args.expname)
save_model_dir = os.path.join(output_csv_dir, 'models')

def main():
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

    # testing
    expname = args.expname.split('/')[-1]
    print('\n' + expname + ': TESTING!')
    train_set = os.path.basename(args.train_list).split('.')[0]
    val_set = os.path.basename(args.val_list).split('.')[0]
    test_set = os.path.basename(args.test_list).split('.')[0]

    if cmd_args.average:
        record = pd.read_csv(os.path.join(output_csv_dir, 'output.csv'), index_col=0)
        sorted_r = record.sort_values('f1', ascending=False)

        model_list = list(sorted_r['epoch_num'].astype(int))[:3]
        df = pd.DataFrame(columns=['exp', 'train', 'val', 'test', 'test_loss', 'test_acc', 'f1'])
        for idx, epoch in enumerate(model_list):
            model = load_checkpoint(model, save_model_dir, f'epoch{epoch}_checkpoint.pth.tar')
            test_loss, test_acc, f1, _ = test(output_csv_dir, test_loader, model, criterion, device, ep_idx=idx+1)

            df.loc[len(df)] = [expname, train_set, val_set, test_set, test_loss, test_acc, f1]

        output_csv_file = os.path.join(output_csv_dir, 'test_acc.csv')
        df.to_csv(output_csv_file, index=False)
    else:
        df = pd.DataFrame(columns=['exp', 'train', 'val', 'test', 'test_loss', 'test_acc', 'f1'])
        model = load_checkpoint(model, save_model_dir, 'model_best_f1.pth.tar')
        test_loss, test_acc, f1, _ = test(output_csv_dir, test_loader, model, criterion, device)

        df.loc[len(df)] = [expname, train_set, val_set, test_set, test_loss, test_acc, f1]
        output_csv_file = os.path.join(output_csv_dir, 'test_f1.csv')
        df.to_csv(output_csv_file, index=False)


def create_model(num_classes):
    if args.network == 100:
        model = resnet.resnet18(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    if args.network == 101:
        model = resnet.resnet50(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    elif args.network == 102:
        architecture = os.path.basename(cmd_args.bit_model)
        model = resnetv2.KNOWN_MODELS[architecture.split('.')[0]](head_size=num_classes, zero_head=True)
        model.load_from(np.load(cmd_args.bit_model))
        print(f'Load pre-trained model {cmd_args.bit_model}')
    elif args.network == 103:
        model = microsoftvision.resnet50(pretrained=True)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_classes))
    else:
        print('model not available! Using PyTorch ResNet50 as default')
        model = resnet.resnet50(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def load_checkpoint(model, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def test(out_dir, test_loader, model, criterion, device, ep_idx=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    tbar = tqdm(test_loader, desc='\r')

    model.eval()
    with torch.no_grad():
        preds = []
        gts = []
        names = []
        scores = []

        for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
            inputs = inputs.float()

            inputs, targets = inputs.to(device), targets.to(device)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            [prec1, ] = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            score = F.softmax(outputs, dim=1)
            pred = score.data.max(1)[1]

            scores.extend(score.tolist())
            preds.extend(pred.tolist())
            gts.extend(targets.tolist())
            names.extend(image_path)

            tbar.set_description('\r %s Loss: %.3f | Top1: %.3f' % ('Test Stats', losses.avg, top1.avg))

        print(confusion_matrix(gts, preds))
        f1s = f1_score(gts, preds, average=None)
        f1_avg = sum(f1s) / len(f1s)

        if ep_idx is None:
            epoch_summary(out_dir, names, scores, preds, gts)
        else:
            epoch_summary(out_dir, names, scores, preds, gts, ep_idx=ep_idx)

    return losses.avg, top1.avg, f1_avg, f1s


def epoch_summary(out_dir, names, scores, preds, targets, ep_idx=None):
    if ep_idx is None:
        csv_file_name = os.path.join(out_dir, 'epoch_test_f1.csv')
    else:
        csv_file_name = os.path.join(out_dir, f'top{ep_idx}_epoch_test_f1.csv')

    if args.dataset == 'renal':
        data =[[name, pred, sum(score[1:4]), target] for name, pred, score, target in zip(names, preds, scores, targets)]
        df = pd.DataFrame(data, columns=['image', 'prediction', 'sclerosis_score', 'target'])
    else:
        data =[[name, pred, target] for name, pred, target in zip(names, preds, targets)]
        df = pd.DataFrame(data, columns=['image', 'prediction', 'target'])

    df.to_csv(csv_file_name)


if __name__ == '__main__':
    main()