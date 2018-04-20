#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### "/home/lilin/my_code/ucf101_split_2_rgb_flow_models/ucf101_split_2_rgb_seg124_3dropout_stage_2_imagenet_lr_0.01_iter_2700.caffemodel" accuracy = 0.864
### /home/lilin/my_code/ucf101_split_2_rgb_flow_models/ucf101_split_2_rgb_seg124_3dropout_stage_2_imagenet_lr_0.001_iter_1200.caffemodel accuracy = 0.888
$TOOLS/caffe train --solver=models/ucf101/rgb_seg124_3dropout_split_2_imagenet_solver.prototxt  \
	--weights="/home/lilin/my_code/ucf101_split_2_rgb_flow_models/ucf101_split_2_rgb_seg124_3dropout_stage_2_imagenet_lr_0.001_iter_1200.caffemodel"