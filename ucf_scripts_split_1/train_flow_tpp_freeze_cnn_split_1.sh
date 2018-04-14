#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_freeze_cnn_lr_0.001_iter_1800.caffemodel accuracy = 0.882
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_freeze_cnn_lr_0.0001_iter_900.caffemodel accurayc= 0.883

mpirun -np 2 \
$TOOLS/caffe train --solver=models/ucf101/flow_tpp_freeze_cnn_split_1_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_freeze_cnn_lr_0.0001_iter_900.caffemodel"