import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from data_loader import DataLoader
from tqdm import tqdm
from util import parse_args, create_model
from utils import AverageMeter, accuracy


def main():
    args = parse_args('test')
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
    model = create_model(num_classes, args).to(device)

    criterion = nn.CrossEntropyLoss()

    # testing
    expname = args.expname.split('/')[-1]
    print('\n' + expname + ': TESTING!')
    train_set = os.path.basename(args.train_list).split('.')[0]
    val_set = os.path.basename(args.val_list).split('.')[0]
    test_set = os.path.basename(args.test_list).split('.')[0]

    if args.average:
        record = pd.read_csv(os.path.join(args.output_csv_dir, 'output.csv'), index_col=0)
        sorted_r = record.sort_values('f1', ascending=False)

        model_list = list(sorted_r['epoch_num'].astype(int))[:3]
        df = pd.DataFrame(columns=['exp', 'train', 'val', 'test', 'test_loss', 'test_acc', 'f1'])
        for idx, epoch in enumerate(model_list):
            model = load_checkpoint(model, os.path.join(args.save_model_dir, f'epoch{epoch}_checkpoint.pth.tar'))
            test_loss, test_acc, f1, _ = test(test_loader, model, criterion, device, args, epoch=idx+1)

            df.loc[len(df)] = [expname, train_set, val_set, test_set, test_loss, test_acc, f1]

        output_csv_file = os.path.join(args.output_csv_dir, 'test_acc.csv')
        df.to_csv(output_csv_file, index=False)
    else:
        df = pd.DataFrame(columns=['exp', 'train', 'val', 'test', 'test_loss', 'test_acc', 'f1'])
        model = load_checkpoint(model, os.path.join(save_model_dir, 'model_best_f1.pth.tar'))
        test_loss, test_acc, f1, _ = test(test_loader, model, criterion, device, args)

        df.loc[len(df)] = [expname, train_set, val_set, test_set, test_loss, test_acc, f1]
        output_csv_file = os.path.join(output_csv_dir, 'test_f1.csv')
        df.to_csv(output_csv_file, index=False)


def test(test_loader, model, criterion, device, args, epoch=None):
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

        epoch_summary(names, scores, preds, gts, args, epoch=epoch)

    return losses.avg, top1.avg, f1_avg, f1s


def epoch_summary(names, scores, preds, targets, args, epoch=None):
    if epoch is None:
        csv_file_name = os.path.join(args.output_csv_dir, 'epoch_test_f1.csv')
    else:
        csv_file_name = os.path.join(args.output_csv_dir, f'top{epoch}_epoch_test_f1.csv')

    if args.dataset == 'renal':
        data =[[name, pred, sum(score[1:4]), target] for name, pred, score, target in zip(names, preds, scores, targets)]
        df = pd.DataFrame(data, columns=['image', 'prediction', 'sclerosis_score', 'target'])
    else:
        data =[[name, pred, target] for name, pred, target in zip(names, preds, targets)]
        df = pd.DataFrame(data, columns=['image', 'prediction', 'target'])

    df.to_csv(csv_file_name)


if __name__ == '__main__':
    main()