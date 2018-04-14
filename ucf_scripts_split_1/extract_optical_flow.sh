#!/usr/bin/env bash

SRC_FOLDER=$1
OUT_FOLDER=$2
NUM_WORKER=$3
FLOW_TYPE=$4
echo "Extracting optical flow from videos in folder: ${SRC_FOLDER}"
python tools/build_of.py ${SRC_FOLDER} ${OUT_FOLDER} --num_worker ${NUM_WORKER} --flow_type ${FLOW_TYPE} --new_width 340 --new_height 256 2>local/errors.log
echo "here"
