#!/bin/bash

for exp in "none" "cutmix" "circlemix"
  do for i in {1..5}
    do
      python ../test.py ../renal/config/efficientb0_${exp}_fold${i}.yaml efficientnet-b0
    done
  done