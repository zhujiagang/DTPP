#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-tpp-net/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

$TOOLS/caffe train --solver=models/hmdb51/kinetics_rgb_tpp_p124_split_3_solver.prototxt  \
   --weights=