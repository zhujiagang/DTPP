#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
$TOOLS/caffe train --solver=models/hmdb51/rgb_seg124_3dropout_split_1_solver.prototxt  \
   --weights="/home/lilin/my_code/ucf101_split_1_rgb_flow_models/bn_inception_rgb_init.caffemodel"