#!/bin/bash

cd ..
for exp in "resnet50_torch"
  do for i in {1..5}
    do
      python predict.py --config renal/config/${exp}_fold${i}.yaml --average True
    done
  done

for exp in "resnet50_bit-s"
  do for i in {1..5}
    do
      python predict.py --config renal/config/${exp}_fold${i}.yaml --bit_model /Data/luy8/glomeruli/models/pretrained_models/BiT-S-R50x1.npz --average True
    done
  done

for exp in "resnet50_bit-m"
  do for i in {1..5}
    do
      python predict.py --config renal/config/${exp}_fold${i}.yaml --bit_model /Data/luy8/glomeruli/models/pretrained_models/BiT-M-R50x1.npz --average True
    done
  done


for exp in "resnet101_torch"
  do for i in {1..5}
    do
      python predict.py --config renal/config/${exp}_fold${i}.yaml --average True
    done
  done

for exp in "resnet101_bit-s"
  do for i in {1..5}
    do
      python predict.py --config renal/config/${exp}_fold${i}.yaml --bit_model /Data/luy8/glomeruli/models/pretrained_models/BiT-S-R101x1.npz --average True
    done
  done

for exp in "resnet101_bit-m"
  do for i in {1..5}
    do
      python predict.py --config renal/config/${exp}_fold${i}.yaml --bit_model /Data/luy8/glomeruli/models/pretrained_models/BiT-M-R101x1.npz --average True
    done
  done

