#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-scale/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

$TOOLS/caffe train --solver=models/ucf101/tsn_bn_inception_flow_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split1_tsn_flow_bn_inception_scale_iter_8700.caffemodel"