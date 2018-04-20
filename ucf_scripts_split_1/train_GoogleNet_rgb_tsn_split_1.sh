#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

### /home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_GoogleNet_tsn_rgb_lr_0.0001_iter_3000.caffemodel accuracy = 0.808
$TOOLS/caffe train --solver=models/ucf101/GoogleNet_tsn_rgb_split_1_solver.prototxt  \
   --weights="/home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_GoogleNet_tsn_rgb_lr_0.0001_iter_3000.caffemodel"