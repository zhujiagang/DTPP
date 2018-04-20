#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_seg124_3dropout_imagenet_stage_2_lr_0.01_iter_5400.caffemodel accuracy = 0.84
### /home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_seg124_3dropout_imagenet_stage_2_lr_0.001_iter_3300.caffemodel
### /home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_seg124_3dropout_imagenet_stage_2_lr_0.0001_iter_2700.caffemodel
mpirun -np 2 \
$TOOLS/caffe train --solver=models/ucf101/flow_seg124_3dropout_imagenet_solver.prototxt  \
   --snapshot="/home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_seg124_3dropout_imagenet_stage_2_lr_0.00001_iter_1500.solverstate"