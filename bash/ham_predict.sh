#!/bin/bash

for exp in "circlemix" "none" "cutmix"
  do for i in {1..5}
    do
      python ../predict.py ../skin/config/efficientb0_${exp}_fold${i}.yaml efficientnet-b0
    done
  done