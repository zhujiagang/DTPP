#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net-pooling/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /home/lilin/my_code/hmdb51_split_3_rgb_flow_models/hmdb51_split_3_rgb_tpp_p1248_lr_0.01_iter_672.caffemodel accuracy = 0.578
### /home/lilin/my_code/hmdb51_split_3_rgb_flow_models/hmdb51_split_3_rgb_tpp_p1248_lr_0.001_iter_1568.caffemodel accuracy = 0.61
$TOOLS/caffe train --solver=models/hmdb51/rgb_tpp_p1248_split_3_solver.prototxt  \
   --weights="/home/lilin/my_code/hmdb51_split_3_rgb_flow_models/hmdb51_split_3_rgb_tpp_p1248_lr_0.001_iter_1568.caffemodel"