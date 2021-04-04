#!/bin/bash

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_cutmix_fold1.yaml efficientnet-b0

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_cutmix_fold2.yaml efficientnet-b0

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_cutmix_fold3.yaml efficientnet-b0

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_cutmix_fold4.yaml efficientnet-b0

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_cutmix_fold5.yaml efficientnet-b0

