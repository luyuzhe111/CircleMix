#!/bin/bash

for exp in "cutmix" "circlemix" "none"
  do
    python ../train.py ../skin/config/efficientb0_${exp}_train_val.yaml efficientnet-b0
  done