#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

### /home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_rgb_tpp_kinetics_lr_0.01_iter_336.caffemodel accuracy = 0.658
### /home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_rgb_tpp_kinetics_lr_0.001_iter_224.caffemodel accuracy = 0.681

$TOOLS/caffe train --solver=models/hmdb51/kinetics_rgb_tpp_p124_split_2_solver.prototxt  \
   --snapshot="/home/lilin/my_code/hmdb51_split_2_rgb_flow_models/hmdb51_split_2_rgb_tpp_kinetics_snapshot_lr_0.00001_iter_224.solverstate"