#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /home/lilin/my_code/ucf101_split_3_rgb_flow_models/ucf101_split_3_flow_seg124_bilinear_lr_0.001_iter_3900.caffemodel accuracy = 0.9193
### /home/lilin/my_code/ucf101_split_3_rgb_flow_models/ucf101_split_3_flow_seg124_bilinear_lr_0.0001_iter_300.caffemodel accuracy = 0.92
mpirun -np 2 \
$TOOLS/caffe train --solver=models/ucf101/flow_seg124_bilinear_split_3_solver.prototxt \
	--weights="/home/lilin/my_code/ucf101_split_3_rgb_flow_models/ucf101_split_3_flow_seg124_bilinear_lr_0.0001_iter_300.caffemodel"