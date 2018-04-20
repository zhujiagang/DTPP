#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_rgb_tpp_delete_dropout_lr_0.01_iter_1344.caffemodel accuracy = 0.577
### /home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_rgb_tpp_delete_dropout_lr_0.01_iter_1904.caffemodel accuracy = 0.59
### /home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_rgb_tpp_delete_dropout_lr_0.001_iter_448.caffemodel accuracy = 0.6

$TOOLS/caffe train --solver=models/hmdb51/rgb_tpp_delete_dropout_split_2_solver.prototxt  \
   --weights="/home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_rgb_tpp_delete_dropout_lr_0.001_iter_448.caffemodel"