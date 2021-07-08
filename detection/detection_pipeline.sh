#!/bin/bash

threshold=0.01
loop=2

if [ "$loop" -eq 1 ]; then
  result_dir="pipeline/thd_${threshold}_result"

elif [ "$loop" -eq 2 ]; then
  result_dir="pipeline/thd_${threshold}_result_${loop}"
fi

python generate_patches.py ${loop} ${threshold}
python filter_patches.py --input ${result_dir} --config ../renal/config/resnet50_0.25_noisy_fold1.yaml --setting resnet50_bit-s --bit_model ../models/pretrained_models/BiT-S-R50x1.npz

python filter_xml.py ${loop} ${threshold}
python evaluate_xml.py ${loop} ${threshold}