#DTPP

This repository holds the codes and models for the paper
 
> 
**End-to-end Video-level Representation Learning for Action Recognition**,
Jiagang Zhu, Wei Zou, Zheng Zhu,
*ICPR 2018*, Beijing, China.
>
[[Arxiv Preprint](https://arxiv.org/abs/1711.04161)]

Below is the guidance to reproduce the reported results and explore more.

# Contents
* [Usage Guide](#usage-guide)
  * [Prerequisites](#prerequisites)
  * [Code & Data Preparation](#code--data-preparation)
    * [Get the code](#get-the-code)
    * [Get the videos](#get-the-videos)
    * [Get trained models](#get-trained-models)
  * [Extract Frames and Optical Flow Images](#extract-frames-and-optical-flow-images)
  * [Testing Provided Models](#testing-provided-models)
    * [Get reference models](#get-reference-models)
    * [Video-level testing](#video-level-testing)
  * [Training DTPP](#training-DTPP)
    * [Construct file lists for training and validation](#construct-file-lists-for-training-and-validation)
    * [Get initialization models](#get-initialization-models)
    * [Start training](#start-training)
    * [Config the training process](#config-the-training-process)
* [Other Info](#other-info)
  * [Citation](#citation)
  * [Related Projects](#related-projects)
  * [Contact](#contact)

----
# Usage Guide

## Prerequisites
[[back to top](#DTPP)]

There are a few dependencies to run the code. The major libraries we use are

- [Caffe][caffe]
- [dense_flow][df]

The codebase is written in Python. We recommend the [Anaconda][anaconda] Python distribution. Matlab scripts are provided for some critical steps like video-level testing.

## Code & Data Preparation

### Get the code
[[back to top](#DTPP)]

Use git to clone this repository and its submodules
```
git clone --recursive https://github.com/zhujiagang/DTPP.git
```

Then run the building scripts to build the libraries.

```
bash build_all.sh
```
It will build Caffe and dense_flow. Since we need OpenCV to have Video IO, which is absent in most default installations, it will also download and build a local installation of OpenCV and use its Python interfaces.

Note that to run training with multiple GPUs, one needs to enable MPI support of Caffe. To do this, run

```
MPI_PREFIX=<root path to openmpi installation> bash build_all.sh MPI_ON
```

### Get the videos
[[back to top](#DTPP)]

We experimented on two mainstream action recognition datasets: [UCF-101][ucf101] and [HMDB51][hmdb51]. Videos can be downloaded directly from their websites.
After download, please extract the videos from the `rar` archives.
- UCF101: the ucf101 videos are archived in the downloaded file. Please use `unrar x UCF101.rar` to extract the videos.
- HMDB51: the HMDB51 video archive has two-level of packaging.
The following commands illustrate how to extract the videos.
```
mkdir rars && mkdir videos
unrar x hmdb51-org.rar rars/
for a in $(ls rars); do unrar x "rars/${a}" videos/; done;
```

### Get trained models
[[back to top](#DTPP)]

We provided the trained model weights in Caffe style, consisting of specifications in Protobuf messages, and model weights.
In the codebase we provide the model spec for UCF101 and HMDB51.
The model weights can be downloaded by running the script

```
bash scripts/get_reference_models.sh
```

## Extract Frames and Optical Flow Images
[[back to top](#DTPP)]

To run the training and testing, we need to decompose the video into frames. Also the DTPP need optical flow images for input.
 
These can be achieved with the script `scripts/extract_optical_flow.sh`. The script has three arguments
- `SRC_FOLDER` points to the folder where you put the video dataset
- `OUT_FOLDER` points to the root folder where the extracted frames and optical images will be put in
- `NUM_WORKER` specifies the number of GPU to use in parallel for flow extraction, must be larger than 1

The command for running optical flow extraction is as follows

```
bash scripts/extract_optical_flow.sh SRC_FOLDER OUT_FOLDER NUM_WORKER
```

It will take from several hours to several days to extract optical flows for the whole datasets, depending on the number of GPUs.  

## Testing Provided Models

### Get reference models
[[back to top](#DTPP)]

To help reproduce the results reported in the paper, we provide reference models trained by us for instant testing. Please use the following command to get the reference models.

```
bash scripts/get_reference_models.sh
```

### Video-level testing
[[back to top](#DTPP)]

We provide a Python framework to run the testing. For the benchmark datasets, we will measure average accuracy on the testing splits. We also provide the facility to analyze a single video.

Generally, to test on the benchmark dataset, we can use the scripts `eval_net.py` and `eval_scores.py`.

For example, to test the reference rgb stream model on split 1 of ucf 101 with 4 GPUs, run
```
python tools/eval_net.py ucf101 1 rgb FRAME_PATH \
 models/ucf101/tsn_bn_inception_rgb_deploy.prototxt models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel \
 --num_worker 4 --save_scores SCORE_FILE
```
where `FRAME_PATH` is the path you extracted the frames of UCF-101 to and `SCORE_FILE` is the filename to store the extracted scores.

One can also use cached score files to evaluate the performance. To do this, issue the following command

```
python tools/eval_scores.py SCORE_FILE
```

The more important function of `eval_scores.py` is to do modality fusion.
For example, once we got the scores of rgb stream in `RGB_SCORE_FILE` and flow stream in `FLOW_SCORE_FILE`.
The fusion result with weights of `1:1.5` can be achieved with

```
python tools/eval_scores.py RGB_SCORE_FILE FLOW_SCORE_FILE --score_weights 1 1.5
```

To view the full help message of these scripts, run `python eval_net.py -h` or `python eval_scores.py -h`. 

## Training DTPP
[[back to top](#DTPP)]

Training TSN is straightforward. We have provided the necessary model specs, solver configs, and initialization models.
To achieve optimal training speed,
we strongly advise you to turn on the parallel training support in the Caffe toolbox using following build command
```
MPI_PREFIX=<root path to openmpi installation> bash build_all.sh MPI_ON
```

where `root path to openmpi installation` points to the installation of the OpenMPI, for example `/usr/local/openmpi/`.

### Construct file lists for training and validation
[[back to top](#DTPP)]

The data feeding in training relies on `VideoDataLayer` in Caffe.
This layer uses a list file to specify its data sources.
Each line of the list file will contain a tuple of extracted video frame path, video frame number, and video groundtruth class.
A list file looks like
```
video_frame_path 100 10
video_2_frame_path 150 31
...
```
To build the file lists for all 3 splits of the two benchmark dataset, we have provided a script.
Just use the following command
```
bash scripts/build_file_list.sh ucf101 FRAME_PATH
```
and
```
bash scripts/build_file_list.sh hmdb51 FRAME_PATH
```
The generated list files will be put in `data/` with names like `ucf101_flow_val_split_2.txt`.

### Get initialization models
[[back to top](#DTPP)]

We have built the initialization model weights for both rgb and flow input.
The flow initialization models implements the cross-modality training technique in the paper.
To download the model weights, run
```
bash get_init_models.sh
bash get_kinetics_pretraining_models.sh
```

### Start training
[[back to top](#DTPP)]

Once all necessities ready, we can start training DTPP.
Fro example, if we want to train on HMDB51.
For example, the following command runs training on HMDB51 with rgb input, with its weights initialized by ImageNet pretraining.
```
bash hmdb_scripts_split_1/train_rgb_tpp_delete_dropout_split_1.sh
```
the training will run with default settings on 1 GPUs.

The learned model weights will be saved in `snapshot/`.
The aforementioned testing process can be used to evaluate them.

 
#Other Info
[[back to top](#DTPP)]

## Citation
Please cite the following paper if you feel this repository useful.
```
@inproceedings{ICPR2018,
  author    = {Jiagang Zhu and
               Wei Zou and
               Zheng Zhu},
  title     = {End-to-end Video-level Representation Learning for Action Recognition},
  booktitle   = {ICPR},
  year      = {2018},
}
```


## Contact
For any question, please contact
```
Jiagang Zhu: zhujiagang2015@ia.ac.cn
```

[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow
[anaconda]:https://www.continuum.io/downloads
[tsn_site]:http://yjxiong.me/others/tsn/
