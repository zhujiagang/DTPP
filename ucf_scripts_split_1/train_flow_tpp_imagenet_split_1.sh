#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_imagenet_lr_0.01_iter_2100.caffemodel accuracy = 0.82
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_imagenet_lr_0.001_iter_2700.caffemodel accuracy = 0.8739
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_imagenet_lr_0.0001_iter_300.caffemodel accuracy = 0.876
mpirun -np 2 \
$TOOLS/caffe train --solver=models/ucf101/flow_tpp_delete_dropout_split_1_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_imagenet_lr_0.0001_iter_300.caffemodel"