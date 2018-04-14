#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

$TOOLS/caffe train --solver=models/ucf101/rgb_tpp_dropout_0.9_split_1_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_seg124_dropout_0.9_lr_0.001_iter_300.caffemodel"