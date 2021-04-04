#!/bin/bash

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_none_fold1.yaml efficientnet-b0

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_none_fold2.yaml efficientnet-b0

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_none_fold3.yaml efficientnet-b0

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_none_fold4.yaml efficientnet-b0

python /Data/luy8/centermix/train.py /Data/luy8/centermix/config_ham/Efficientb0_none_fold5.yaml efficientnet-b0
