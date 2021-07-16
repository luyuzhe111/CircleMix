#!/bin/bash

threshold=0.01
loop=2
generate_patch=false

if [ "$loop" -eq 1 ]; then
  result_dir="pipeline/thd_${threshold}_result"

elif [ "$loop" -eq 2 ]; then
  result_dir="pipeline/thd_${threshold}_result_${loop}"
fi

if $generate_patch; then
  python generate_patches.py ${loop} ${threshold}
fi

#python filter_patches.py --input ${result_dir} --config ../renal/config/resnet50_bit-m_fold1.yaml --bit_model models/pretrained_models/BiT-M-R50x1.npz

#python filter_xml.py ${loop} ${threshold}
python evaluate_xml.py ${loop} ${threshold}