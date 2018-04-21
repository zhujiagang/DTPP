# DTPP

This repository holds the codes and models for the paper
 
> 
**End-to-end Video-level Representation Learning for Action Recognition**,
Jiagang Zhu, Wei Zou, Zheng Zhu,
*ICPR 2018*, Beijing, China.

>

[[Arxiv Preprint](https://arxiv.org/abs/1711.04161)]

We follow the guidance provided by [TSN][tsn] to prepare the data. Please refer to the [TSN][tsn] repository for guidance. Here we only provide the additional training details of DTPP.

# Usage Guide

## Code & Data Preparation

### Get the code


Use git to clone this repository and its submodules
```
git clone --recursive https://github.com/zhujiagang/DTPP.git
```
### Compile Caffe 
cd lib/
compile caffe-tpp-net/ with cmake and openmpi according to TSN for training models
and compile caffe-tpp-net-python/ with python interface for evaluating models with python script.

### Get initialization models

We have built the initialization model weights for both rgb and flow input.
The flow initialization models implements the cross-modality training technique in the paper.
To download the model weights, run
```
bash get_init_models.sh
bash get_kinetics_pretraining_models.sh
```

### Start training
[[back to top](#dtpp)]

Once all necessities ready, we can start training DTPP.
Fro example, if we want to train on HMDB51.
For example, the following command runs training on HMDB51 with rgb input, with its weights initialized by ImageNet pretraining.
```
bash hmdb_scripts_split_1/train_rgb_tpp_delete_dropout_split_1.sh
```
The learned model weights will be saved in `snapshot/`.
The aforementioned testing process can be used to evaluate them.

 
#Other Info

## Citation
Please cite the following paper if you feel this repository useful.
```
@inproceedings{DTPP2018ICPR,
  author    = {Jiagang Zhu and
               Wei Zou and
               Zheng Zhu},
  title     = {End-to-end Video-level Representation Learning for Action Recognition},
  booktitle   = {ICPR},
  year      = {2018},
}

@inproceedings{TSN2016ECCV,
  author    = {Limin Wang and
               Yuanjun Xiong and
               Zhe Wang and
               Yu Qiao and
               Dahua Lin and
               Xiaoou Tang and
               Luc {Val Gool}},
  title     = {Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
  booktitle   = {ECCV},
  year      = {2016},
}
```


## Contact
For any question, please contact
```
Jiagang Zhu: zhujiagang2015@ia.ac.cn
```
[tsn]:https://github.com/yjxiong/temporal-segment-networks#temporal-segment-networks-tsn
