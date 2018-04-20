#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

### /home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_seg124_3dropout_sec_lr_0.001_iter_6300.caffemodel accuracy = 0.887
### /home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_seg124_3dropout_sec_lr_0.0001_iter_600.caffemodel accuracy = 0.888
mpirun -np 2 \
$TOOLS/caffe train --solver=models/ucf101/flow_seg124_3dropout_sec_solver.prototxt \
	--weights="/home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_seg124_3dropout_sec_lr_0.0001_iter_600.caffemodel"