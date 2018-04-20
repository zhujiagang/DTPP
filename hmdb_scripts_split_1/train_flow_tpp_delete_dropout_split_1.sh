#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
### /home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_flow_tpp_delete_dropout_lr_0.001_iter_448.caffemodel accuracy = 0.646
### /home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672.caffemodel accuracy = 0.659
mpirun -np 2 \
$TOOLS/caffe train --solver=models/hmdb51/flow_tpp_delete_dropout_split_1_solver.prototxt  \
   --weights="/home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672.caffemodel"