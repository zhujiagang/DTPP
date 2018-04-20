#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

$TOOLS/caffe train --solver=models/ucf101/rgb_seg124_3dropout_split_2_solver.prototxt  \
   --weights="/home/lilin/my_code/ucf101_split_2_rgb_flow_models/ucf101_split_2_tsn_rgb_reference_bn_inception.caffemodel"