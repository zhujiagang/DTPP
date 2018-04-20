#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

mpirun -np 2 \
$TOOLS/caffe train --solver=models/ucf101/vgg_rgb_seg124_fc_solver.prototxt  \
   --weights="/home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_vgg_rgb_seg124_fc_lr_0.01_iter_2700.caffemodel"