#!/bin/bash

for exp in "none" "cutmix" "circlemix"
  do
    python ../train.py ../skin/config/efficientb0_${exp}_train_val.yaml efficientnet-b0
  done