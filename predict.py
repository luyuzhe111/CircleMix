import os
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd

from data_loader import DataLoader
from tqdm import tqdm
from util import parse_args, create_model, load_checkpoint
from utils import AverageMeter


def main():
    args = parse_args('predict')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_set = DataLoader(args.ext_test, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print('\nPredict with ' + args.checkpoint.split('/')[-3])

    print("==> creating model")
    num_classes = args.num_classes
    model = create_model(num_classes, args)
    model = load_checkpoint(model, args.checkpoint).to(device)

    predict(test_loader, model, device, args)


def predict(test_loader, model, device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    tbar = tqdm(test_loader, desc='\r')
    end = time.time()

    model.eval()
    with torch.no_grad():
        preds = []
        gts = []
        names = []
        scores = []

        for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            score = F.softmax(outputs, dim=1)
            pred = score.data.max(1)[1]

            scores.extend(score.tolist())
            preds.extend(pred.tolist())
            gts.extend(targets.tolist())
            names.extend(image_path)

        if args.dataset == 'renal':
            data = [[name, pred, sum(score[1:4]), target] for name, pred, score, target in zip(names, preds, scores, gts)]
            df = pd.DataFrame(data, columns=['image', 'prediction', 'sclerosis_score', 'target'])

        else:
            data = [[name, pred, target] for name, pred, target in zip(names, preds, gts)]
            df = pd.DataFrame(data, columns=['image', 'prediction', 'target'])

        df.to_csv(os.path.join(args.output_csv_dir, 'predict_f1.csv'), index=False)


if __name__ == '__main__':
    main()




