#!/bin/bash

for exp in "none" "cutmix" "circlemix"
  do
    python ../predict.py ../skin/config/efficientb0_${exp}_train_val.yaml efficientnet-b0
  done