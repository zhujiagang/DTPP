import argparse
import os
import sys
import math
import cv2
import glob
import fnmatch
import os
import random
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix

from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file

from pyActionRecog.utils.video_funcs import default_aggregation_func


# load split file
class_files = glob.glob('data/hmdb51_splits/*split*.txt')

# load class list
class_list = [x.strip() for x in open('data/hmdb51_splits/class_list.txt')]
class_dict = {x: i for i, x in enumerate(class_list)}

def parse_class_file(filename):
    # parse filename parts
    filename_parts = filename.split('/')[-1][:-4].split('_')
    split_id = int(filename_parts[-1][-1])
    class_name = '_'.join(filename_parts[:-2])

    # parse class file contents
    contents = [x.strip().split() for x in open(filename).readlines()]
    train_videos = [ln[0][:-4] for ln in contents if ln[1] == '1']
    test_videos = [ln[0][:-4] for ln in contents if ln[1] == '2']

    return class_name, split_id, train_videos, test_videos

class_info_list = map(parse_class_file, class_files)

splits = []
for i in xrange(1, 4):
    train_list = [
        (vid, class_dict[cls[0]]) for cls in class_info_list for vid in cls[2] if cls[1] == i
    ]
    test_list = [
        (vid, class_dict[cls[0]]) for cls in class_info_list for vid in cls[3] if cls[1] == i
    ]
    splits.append((train_list, test_list))
    file = open('data/hmdb51_splits/testlist0' + str(i) + '.txt', 'w')
    for ii in range(len(test_list)):
        xx = test_list[ii][0]
        file.write(str(xx)+ ".avi")
        file.write('\n')

    file.close()

