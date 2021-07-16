import argparse
from easydict import EasyDict as edict
import os
import yaml
import models.resnet as resnet
import models.resnetv2 as resnetv2
import microsoftvision
import numpy as np
import torch
import torch.nn as nn

def parse_args(split):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='configuration file')
    parser.add_argument('--bit_model', default=None, help='BiT model')
    parser.add_argument('--average', default=False, type=bool, help='whether to average top3 models during testing')
    parser.add_argument('--input', default=None, help='list of patches to be filtered')

    opt = parser.parse_args()
    config_dir = opt.config
    config_file = os.path.basename(config_dir)
    print(f'{split} with ' + config_file)

    with open(config_dir, 'r') as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    if split == 'train':
        args.expname = config_file.split('.yaml')[0]
        args.optimizer = 'ADAM'
        args.resample = True
        args.loss = 'Focal'
        args.weighted_loss = False

    elif split == 'predict':
        args.expname = config_file.split('.yaml')[0]
        if args.dataset == 'ham':
            args.ext_test = os.path.abspath('skin/json/test.json')
        elif args.dataset == 'renal':
            args.ext_test = os.path.abspath('renal/json/test.json')
        else:
            raise ValueError('dataset does not exist')

    args.expname = config_file.split('.yaml')[0]
    assert split in ['train', 'test', 'predict', 'filter'], f'unknown split: {split}'

    args.update(edict(vars(opt)))

    args.output_csv_dir = os.path.abspath(f'{args.output_csv_dir}/{args.expname}')
    os.makedirs(args.output_csv_dir, exist_ok=True)

    args.log_dir = os.path.abspath(f'{args.output_csv_dir}/log')
    os.makedirs(args.log_dir, exist_ok=True)

    args.save_model_dir = os.path.abspath(f'{args.output_csv_dir}/models')
    os.makedirs(args.save_model_dir, exist_ok=True)

    return args


def create_model(num_classes, args):
    if args.network == 100:
        model = resnet.resnet18(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    elif args.network == 101:
        model = resnet.resnet50(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    elif args.network == 102:
        architecture = os.path.basename(args.bit_model)
        model = resnetv2.KNOWN_MODELS[architecture.split('.')[0]](head_size=num_classes, zero_head=True)
        model.load_from(np.load(args.bit_model))
        print(f'Load pre-trained model {args.bit_model}')
    elif args.network == 103:
        model = resnet.resnet101(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    elif args.network == 104:
        model = microsoftvision.resnet50(pretrained=True)
        model.fc = model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_classes))
    else:
        print('model not available! Using PyTorch ResNet50 as default')
        model = resnet.resnet50(pretrained=args.pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def load_checkpoint(model, filepath):
    assert os.path.isfile(filepath), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model