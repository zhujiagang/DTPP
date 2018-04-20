#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_3dropout_imagenet_stage_2_lr_0.01_iter_1800.caffemodel 0.86
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_3dropout_imagenet_stage_2_lr_0.001_iter_1800.caffemodel accuracy = 0.886
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_3dropout_imagenet_stage_2_lr_0.0001_iter_600.caffemodel accuracy = 0.888
$TOOLS/caffe train --solver=models/ucf101/rgb_seg124_3dropout_imagenet_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_3dropout_imagenet_stage_2_lr_0.0001_iter_600.caffemodel"