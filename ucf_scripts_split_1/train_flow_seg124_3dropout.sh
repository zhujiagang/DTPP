#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-mpi-transpose/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"


mpirun -np 2 \
$TOOLS/caffe train --solver=models/ucf101/flow_seg124_3dropout_solver.prototxt