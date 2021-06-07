#!/bin/bash

network=103

if [ "$network" -eq 101 ]; then
  for setting in "resnet50_randinit" "resnet50_pytorch_pretrain"; do
    for exp in "none" "0.25_noisy"
      do for i in {1..5}
        do
          python ../predict.py --config ../renal/config/resnet50_0.25_${exp}_fold${i}.yaml --setting ${setting} --bit_model /Data/luy8/glomeruli/models/pretrained_models/BiT-M-R50x1.npz
        done
      done
  done
fi

if [ "$network" -eq 102 ]; then
  for setting in "resnet50_bit-s" "resnet50_bit-m"; do
    for exp in "none" "0.25_noisy"; do
      for i in {1..5}; do
            if [[ "$setting" == *"bit-s"* ]]; then
                python ../predict.py --config ../renal/config/resnet50_${exp}_fold${i}.yaml --setting ${setting} --bit_model /Data/luy8/glomeruli/models/pretrained_models/BiT-S-R50x1.npz
            fi

            if [[ "$setting" == *"bit-m"*  ]]; then
                python ../predict.py --config ../renal/config/resnet50_${exp}_fold${i}.yaml --setting ${setting} --bit_model /Data/luy8/glomeruli/models/pretrained_models/BiT-M-R50x1.npz
            fi
      done
    done
  done
fi


if [ "$network" -eq 101 ]; then
  for setting in "resnet50_ms_vision"
    do for exp in "none" "0.25_noisy"
      do for i in {1..5}
        do
          python ../predict.py --config ../renal/config/resnet50_${exp}_fold${i}.yaml --setting ${setting} --bit_model /Data/luy8/glomeruli/models/pretrained_models/BiT-S-R50x1.npz
        done
      done
    done
fi
