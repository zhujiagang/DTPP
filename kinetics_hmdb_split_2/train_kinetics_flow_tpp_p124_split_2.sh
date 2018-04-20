#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"
# sleep 3h
# kill 18653
### /home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_flow_tpp_kinetics_lr_0.01_iter_448.caffemodel accuracy = 0.748
### /home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_flow_tpp_kinetics_lr_0.001_iter_336.caffemodel accuracy = 0.78

$TOOLS/caffe train --solver=models/hmdb51/kinetics_flow_tpp_p124_split_2_solver.prototxt  \
   --weights="/home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_flow_tpp_kinetics_lr_0.001_iter_336.caffemodel"