__author__ = 'zhujiagang'

import os
import glob
import sys
import matplotlib.image as mpimg
from PIL import Image
from pipes import quote
from multiprocessing import Pool, current_process
import numpy as np
from numpy import zeros,arange
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.pyplot import twinx
from math import ceil
import cv2
import argparse

if __name__ == '__main__':

    file_1 = open("/home/lilin/my_code/deeptemporal/data/draw_sword_hmdb51_splits/draw_sword_test_split1.txt")

    file_txt = "/home/lilin/my_code/deeptemporal/data/draw_sword_hmdb51_splits/draw_sword_test_30.txt"
    file = open(file_txt, 'w')
    cnt = 0
    while 1:
        line = file_1.readline()
        if not line:
            break
        item_1 = line.split(' ')[0] #+ '/'
        item_2 = line.split(' ')[1]  # + '/'
        if item_2 is '2':
            file.write(line)


    file.close()


# if __name__ == '__main__':
#
#     file_1 = open("/home/lilin/my_code/deeptemporal/data/newhmdb51_splits/class_list.txt")
#
#     file_txt = "/home/lilin/my_code/deeptemporal/data/newhmdb51_splits/class_list_new.txt"
#     file = open(file_txt, 'w')
#     cnt = 0
#     while 1:
#         line = file_1.readline()
#         if not line:
#             break
#         file.write(str(cnt) + " " + line)
#         cnt += 1
#
#
#     file.close()