#!/bin/bash

python /Data/luy8/centermix/renal/renal_train_sgd.py /Data/luy8/centermix/config_renal/Efficientb0-B_us_aug_centermix_fold1.yaml efficientnet-b0

python /Data/luy8/centermix/renal/renal_train_sgd.py /Data/luy8/centermix/config_renal/Efficientb0-B_us_aug_centermix_fold2.yaml efficientnet-b0

python /Data/luy8/centermix/renal/renal_train_sgd.py /Data/luy8/centermix/config_renal/Efficientb0-B_us_aug_centermix_fold3.yaml efficientnet-b0

python /Data/luy8/centermix/renal/renal_train_sgd.py /Data/luy8/centermix/config_renal/Efficientb0-B_us_aug_centermix_fold4.yaml efficientnet-b0

python /Data/luy8/centermix/renal/renal_train_sgd.py /Data/luy8/centermix/config_renal/Efficientb0-B_us_aug_centermix_fold5.yaml efficientnet-b0
