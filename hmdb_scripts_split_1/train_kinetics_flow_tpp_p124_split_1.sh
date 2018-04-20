#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
# sleep 1h
### /home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_flow_tpp_kinetics_lr_0.01_iter_448.caffemodel accuracy = 0.75
### /home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_flow_tpp_kinetics_lr_0.001_iter_896.caffemodel accuracy = 0.79
### /home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_flow_tpp_kinetics_lr_0.0001_iter_112.caffemodel accuracy  = 0.7937
$TOOLS/caffe train --solver=models/hmdb51/kinetics_flow_tpp_p124_split_1_solver.prototxt  \
   --weights="/home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_flow_tpp_kinetics_lr_0.0001_iter_112.caffemodel"