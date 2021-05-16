import time
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
import sys

import models.resnet as resnet
import models.wideresnet as models
import models.mobilenetv2 as mobilenetv2
import models.senet as senet
from models.efficientnet import EfficientNet
import models.inceptionv4 as inceptionv4
from data_loader import DataLoader
from utils import AverageMeter

from easydict import EasyDict as edict
from argparse import Namespace
import yaml


config_dir = sys.argv[1]
config_file = os.path.basename(config_dir)

with open(config_dir, 'r') as f:
    args = edict(yaml.load(f, Loader=yaml.FullLoader))

args = Namespace(**args)
args.expname = config_file.split('.yaml')[0]

if args.dataset == 'ham':
    args.ext_test_list = '/Data/luy8/glomeruli/skin/json/test.json'
elif args.dataset == 'renal':
    args.ext_test_list = '/Data/luy8/glomeruli/renal/json/all/test.json'
else:
    raise ValueError('dataset does not exist')

output_csv_dir = os.path.join(args.output_csv_dir, args.expname)
if not os.path.exists(output_csv_dir):
    os.mkdir(output_csv_dir)

save_model_dir = os.path.join(output_csv_dir, 'models')
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

pretrained_model = os.path.join(save_model_dir, 'model_best_f1.pth.tar')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(0)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_set = DataLoader(args.ext_test_list, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print('\nTest with ' + pretrained_model.split('/')[-3])

    print("==> creating model")
    num_classes = args.num_classes
    model = create_model(num_classes).to(device)
    model = load_checkpoint(model, pretrained_model)

    predict(output_csv_dir, test_loader, model, device)


def create_model(num_classes):
    if args.network == 101:
        model = models.WideResNet(num_classes=num_classes)
    elif args.network == 102:
        model = resnet.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif args.network == 103:
        model = mobilenetv2.mobilenet_v2(pretrained=True)
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


def load_checkpoint(model, filepath):
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def predict(out_dir, valloader, model, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    tbar = tqdm(valloader, desc='\r')
    end = time.time()

    with torch.no_grad():
        pred_history = []
        target_history = []
        name_history = []
        prob_history = []

        for batch_idx, (inputs, targets, image_path) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            prob = F.softmax(outputs, dim=1)
            pred = prob.data.max(1)[1]
            pred_history = np.concatenate((pred_history, pred.data.cpu().numpy()), axis=0)
            target_history = np.concatenate((target_history, targets.data.cpu().numpy()), axis=0)
            names = [os.path.basename(i).split('.')[0] for i in list(image_path)]
            name_history = np.concatenate((name_history, names), axis=0)

            if batch_idx == 0:
                prob_history = prob.data.cpu().numpy()
            else:
                prob_history = np.concatenate((prob_history, prob.data.cpu().numpy()), axis=0)

        if args.dataset == 'renal':
            prob_pos = prob_history[:, 1] + prob_history[:, 2] + prob_history[:, 3]
            data = np.concatenate((name_history[..., np.newaxis],
                                   prob_history,
                                   prob_pos[..., np.newaxis],
                                   target_history[..., np.newaxis]), axis=1)
            df = pd.DataFrame(data, columns=['image',
                                             'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4', 'pro_pos', 'target'])
        else:
            data = np.concatenate((name_history[..., np.newaxis], pred_history[..., np.newaxis], target_history[..., np.newaxis]), axis=1)
            df = pd.DataFrame(data, columns=['image', 'prediction', 'target'])

        df.to_csv(os.path.join(out_dir, 'predict_f1.csv'), index=False)


if __name__ == '__main__':
    main()




