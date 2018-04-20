__author__ = 'zhujiagang'

import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process
import cv2
import argparse
out_path = ''
str = "/home/zjg/zjg/tsncaffe/UCF-101-vis/v_single/img_00005.jpg"
img = cv2.imread(str)
img = cv2.imread(str)
img[60:260,60:165,:] = 128
cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.imwrite('/home/zjg/zjg/tsncaffe/UCF-101-vis/v_single/img_00005_mask.jpg', img)
cv2.waitKey (0)
cv2.destroyAllWindows()
