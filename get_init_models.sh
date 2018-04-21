#!/usr/bin/env bash

wget -O init_models/bn_inception_rgb_init.caffemodel http://mmlab.siat.ac.cn/tsn_model/bn_inception_spatial_var_init.caffemodel

# ucf101 flow models, 3 splits
wget -O init_models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel.v5
wget -O init_models/ucf101_split_2_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_2_tsn_flow_reference_bn_inception.caffemodel.v5
wget -O init_models/ucf101_split_3_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/ucf101_split_3_tsn_flow_reference_bn_inception.caffemodel.v5

# hmdb51 flow models, 3 splits
wget -O init_models/hmdb51_split_1_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_1_tsn_flow_reference_bn_inception.caffemodel.v5
wget -O init_models/hmdb51_split_2_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_2_tsn_flow_reference_bn_inception.caffemodel.v5
wget -O init_models/hmdb51_split_3_tsn_flow_reference_bn_inception.caffemodel http://mmlab.siat.ac.cn/tsn_model/hmdb51_split_3_tsn_flow_reference_bn_inception.caffemodel.v5
