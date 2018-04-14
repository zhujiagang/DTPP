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
import random
import time
import datetime

sample_num = 16

def get_time_str():
    ts = time.time()
    return ts

out_path = ''


def dump_frames(vid_path):
    import cv2
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in xrange(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print '{} done'.format(vid_name)
    sys.stdout.flush()
    return file_list


def run_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_gpu')+' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
        quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])

    os.system(cmd)
    print '{} {} done'.format(vid_id, vid_name)
    sys.stdout.flush()
    return True


def run_warp_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_warp_gpu')+' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        vid_path, flow_x_path, flow_y_path, dev_id, out_format)

    os.system(cmd)
    print 'warp on {} {} done'.format(vid_id, vid_name)
    sys.stdout.flush()
    return True

def readimg():
    lena = mpimg.imread("/home/zhujiagang/temporal-segment-networks/yulan.jpg")
    im =Image.fromarray(np.uint8(lena*255))
    im.show()

def readimg_2(item_1, item_2, item_3, file):
    vid_path = item_1 + '/'
    cnt_x = 0
    cnt_y = 0
    cnt = 0
    flow_mag = []
    for filename in os.listdir(vid_path):
        if filename.startswith('flow_x_'):
            cnt_x += 1
            flow_x = cv2.imread(vid_path + 'flow_x_{:05d}.jpg'.format(cnt_x), cv2.IMREAD_GRAYSCALE).astype('int32')
            flow_y = cv2.imread(vid_path + 'flow_y_{:05d}.jpg'.format(cnt_x), cv2.IMREAD_GRAYSCALE).astype('int32')
            flow = flow_x * flow_x + flow_y * flow_y
            #print sum(sum(flow_x))
            flow_sum = sum(sum(flow)).astype('float32')
            #print flow_sum
            flow_mag.append(flow_sum)
        if filename.startswith('flow_y_'):
            cnt_y += 1
        if filename.startswith('img_'):
            cnt += 1

    flow_mag = flow_mag/max(flow_mag)
    kk = []
    ii = 0
    while ii < 5 * sample_num:
        #random.seed(10)
        #weight_offset = random.uniform(0, 1 + sum(flow_mag))

        ttt = get_time_str() * 100
        tttt = str(int(ttt))
        tttt = int(tttt[:-7:-1])
        tttt = tttt * random.uniform(1, 2)

        weight_offset = tttt % (1.0 + sum(flow_mag))
        #print weight_offset

        weight_offset_sum = 0
        for i in range(len(flow_mag)):
            weight_offset_sum += flow_mag[i]
            if weight_offset_sum >= weight_offset:
                kk.append(i)
                break
            if i == len(flow_mag) - 1:
                kk.append(i)
        ii += 1

    file.write(item_1 + ' ' + item_2 + ' ' + item_3 + ' ')

    kkk = []
    tall_max = 60
    tall_min = 15
    duration = len(flow_mag) - 1
    for i in range(5 * sample_num):
        kkk.append(kk[i])
        if ( i + 1 ) % 5 == 0:
            kkk = sorted(kkk)
            if duration > tall_max:
                if (kkk[3] - kkk[1]) < tall_max :
                    temp = (tall_max - kkk[3] + kkk[1] ) / 2
                    kkk[3] = kkk[3] + temp
                    kkk[1] = kkk[1] - temp
                    kkk[3] = min(kkk[3], duration)
                    kkk[1] = max(kkk[1], 0)
                    if (kkk[3] - kkk[1]) < tall_max and kkk[3] == duration:
                        while (kkk[3] - kkk[1]) < tall_max and kkk[1] > 0:
                            kkk[1] -= 2
                            kkk[3] -= 1
                        kkk[1] = max(kkk[1], 0)
                        kkk[2] = (kkk[1] + kkk[3]) / 2
                    if (kkk[3] - kkk[1]) < tall_max and kkk[1] == 0:
                        while (kkk[3] - kkk[1]) < tall_max and kkk[3] < duration:
                            kkk[1] += 1
                            kkk[3] += 2
                        kkk[3] = min(kkk[3], duration)
                        kkk[2] = (kkk[1] + kkk[3]) / 2

                if kkk[1] - kkk[0] < tall_min:
                    kkk[0] = max(kkk[1] - tall_min, 0)
                    while (kkk[1] - kkk[0] < tall_min) and kkk[3] < duration - tall_min and kkk[4] < duration:
                        kkk[1] += 1
                        kkk[2] += 1
                        kkk[3] += 1
                        kkk[4] += 1

                if kkk[4] - kkk[3] < tall_min:
                    kkk[4] = min(kkk[3] + tall_min, duration)
                    while (kkk[4] - kkk[3] < tall_min) and kkk[1] > tall_min and kkk[0] > 0:
                        kkk[3] -= 1
                        kkk[2] -= 1
                        kkk[1] -= 1
                        kkk[0] -= 1
            else:
                kkk[0] = 0
                for iii in range(1,5):
                    kkk[iii] = iii * duration / 4

            for item in kkk:
                file.write(str(item) + ' ')
            kkk = []

    file.write('\n')

    assert cnt_y == cnt_x == cnt
    print cnt_x, cnt_y, cnt

def npread():
    #readimg()
    src_path = "/home/zjg/zjg/tsncaffe/UCF-101-result"

    vid_list = glob.glob(src_path+'/*/')
    print len(vid_list)
    file_txt = "importance_sampling_data.txt"
    file = open(file_txt, 'w')
    for index, item in enumerate(vid_list):
        print "start processing: " + item
        readimg_2(item, file)

    file.close()

if __name__ == '__main__':

    file_1 = open("/home/zhujiagang/temporal-segment-networks/data/ucf_train_1.txt")
    file_txt = "/home/zhujiagang/temporal-segment-networks/tools/new_train_data.txt"
    file = open(file_txt, 'w')
    cnt = 0
    while 1:
        line = file_1.readline()
        if not line:
            break
        item_1 = line.split(' ')[0] #+ '/'
        item_2 = line.split(' ')[1]  # + '/'
        item_3 = line.split(' ')[-1].split('\n')[0]

        cnt +=1
        print "start processing {:05d} th item".format(cnt) + item_1
        readimg_2(item_1, item_2, item_3, file)

    file.close()