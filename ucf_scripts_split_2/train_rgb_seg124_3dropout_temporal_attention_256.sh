#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_3dropout_temporal_attention_256_stage_2_lr_0.01_iter_3300.caffemodel accuracy = 0.85

$TOOLS/caffe train --solver=models/ucf101/rgb_seg124_3dropout_temporal_attention_256_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_3dropout_temporal_attention_256_stage_2_lr_0.01_iter_3300.caffemodel"