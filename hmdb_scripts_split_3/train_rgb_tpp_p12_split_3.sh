#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /home/lilin/my_code/hmdb51_split_3_rgb_flow_models/hmdb51_split_3_rgb_tpp_p12_lr_0.01_iter_1344.caffemodel accuracy  = 0.5868
### /home/lilin/my_code/hmdb51_split_3_rgb_flow_models/hmdb51_split_3_rgb_tpp_p12_lr_0.001_iter_224.caffemodel accuracy = 0.607
$TOOLS/caffe train --solver=models/hmdb51/rgb_tpp_p12_split_3_solver.prototxt  \
   --weights="/home/lilin/my_code/hmdb51_split_3_rgb_flow_models/hmdb51_split_3_rgb_tpp_p12_lr_0.001_iter_224.caffemodel"