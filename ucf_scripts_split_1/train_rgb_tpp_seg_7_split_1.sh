#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_tpp_seg_7_lr_0.01_iter_2100.caffemodel accuracy = 0.85
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_tpp_seg_7_lr_0.001_iter_2100.caffemodel accuracy = 0.872
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_tpp_seg_7_lr_0.0001_iter_600.caffemodel accuracy = 0.873
$TOOLS/caffe train --solver=models/ucf101/rgb_tpp_seg_7_split_1_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_tpp_seg_7_lr_0.0001_iter_600.caffemodel"