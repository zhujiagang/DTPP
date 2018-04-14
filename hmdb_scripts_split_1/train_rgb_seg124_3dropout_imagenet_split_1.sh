#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

### /data4/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_rgb_seg124_3dropout_imagenet_lr_0.01_iter_2128.caffemodel accuracy = 0.595
echo "logging to ${LOG_FILE}"
$TOOLS/caffe train --solver=models/hmdb51/rgb_seg124_3dropout_imagenet_split_1_solver.prototxt  \
   --snapshot="/data4/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_rgb_seg124_3dropout_imagenet_lr_0.001_iter_3696.solverstate"