#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
#### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_snapshot300_stage_2_lr_0.1_iter_2100.caffemodel accuracy = 0.697
#### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_snapshot300_stage_2_lr_0.01_iter_900.caffemodel accuracy = 0.8
#### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_snapshot300_stage_2_lr_0.001_iter_750.caffemodel accuracy = 0.823


### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_batch_4_stage_2_lr_0.01_iter_6300.caffemodel accuracy = 0.825
### /data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_batch_4_stage_2_lr_0.001_iter_2175.caffemodel accuracy = 0.86

$TOOLS/caffe train --solver=models/ucf101/rgb_seg124_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_stage_1_lr_0.001_iter_2325.caffemodel"