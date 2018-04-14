#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_seg_4_lr_0.001_iter_1500.caffemodel accuracy = 0.85
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_seg_4_lr_0.0001_iter_1200.caffemodel accuracy = 0.853
mpirun -np 2 \
$TOOLS/caffe train --solver=models/ucf101/flow_tpp_seg_4_split_1_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_seg_4_lr_0.0001_iter_1200.caffemodel"