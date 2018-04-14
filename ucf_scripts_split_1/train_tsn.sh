#!/usr/bin/env bash

DATASET=$1

filename=$(date '+%Y%m%d_%H:%M:%S')
echo $filename

TOOLS=lib/caffe-action/build/install/bin
LOG_FILE=logs/${DATASET}_${filename}.log

echo "logging to ${LOG_FILE}"

#${MPI_BIN_DIR}mpirun -np $N_GPU \
#$TOOLS/caffe train --solver=models/${DATASET}/tsn_bn_inception_${MODALITY}_solver.prototxt  \
#-snapshot models/ucf101_split1_tsn_flow_bn_inception_iter_1000.solverstate 2>&1 | tee ${LOG_FILE}
   #--weights=models/bn_inception_${MODALITY}_init.caffemodel 2>&1 | tee ${LOG_FILE}


#$TOOLS/caffe train --solver=models/ucf101/gating_tsn_solver.prototxt  \
  # --weights models/gating_tsn_prob_5img_split1_iter_2000.caffemodel   2>&1 | tee ${LOG_FILE}
#/usr/local/openmpi/bin/mpirun -np $N_GPU \
$TOOLS/caffe train --solver=models/ucf101/gating_three_2_solver.prototxt  \
   --weights="/data4/lilin/my_code/ucf101_split_3_rgb_flow_models/ucf101_split_3_gating_three_2_iter_1292.caffemodel" 2>&1 | tee ${LOG_FILE}
   #--snapshot=models/ucf101_split1_tsn_${MODALITY}_bn_inception_iter_2000.solverstate 2>&1 | tee ${LOG_FILE}   
   #--weights=models/ucf101_split1_tsn_rgb_bn_inception_iter_3500.caffemodel 2>&1 | tee ${LOG_FILE}