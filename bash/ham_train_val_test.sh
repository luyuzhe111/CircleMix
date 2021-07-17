#!/bin/bash

for exp in "circlemix" "none"  "cutmix"
  do
    python ../predict.py ../skin/config/efficientb0_${exp}_train_val.yaml efficientnet-b0
  done