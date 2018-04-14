__author__ = 'zhujiagang'

import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process

import argparse
out_path = ''

def dump_frames(vid_path, size):
    import cv2
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    #print vid_name
    #print type(vid_name)
    str1 = "images"
    #print type(str1)
    out_full_path = os.path.join(out_path, vid_name, str1)
    #print out_full_path
    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    if not os.path.isdir(out_full_path):
        print "creating folder: "+out_full_path
        os.makedirs(out_full_path)

    file_list = []
    for i in xrange(fcount):
        ret, frame = video.read()
        #assert ret
        if ret and not os.path.isfile('{}/{}.avi_{:06d}.jpg'.format(out_full_path, vid_name, i)):
            imageResize = cv2.resize(frame, size)
            cv2.imwrite('{}/{}.avi_{:06d}.jpg'.format(out_full_path, vid_name, i), imageResize)
            access_path = '{}/{}.avi_{:06d}.jpg'.format(vid_name, vid_name, i)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    #parser.add_argument("src_dir")
    #parser.add_argument("out_dir")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'])
    parser.add_argument("--df_path", type=str, default='./lib/dense_flow/', help='path to the dense_flow toolbox')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='path to the dense_flow toolbox')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU')

    args = parser.parse_args()

    out_path = "/home/zjg/zjg/tsncaffe/UCF-101-result-1/"
    src_path = "/home/zjg/zjg/tsncaffe/UCF-101/"
    num_worker = args.num_worker
    flow_type = args.flow_type
    df_path = args.df_path
    out_format = args.out_format
    ext = args.ext

    new_size = (340, 256)
    print new_size
    NUM_GPU = args.num_gpu

    if not os.path.isdir(out_path):
        print "creating folder: "+out_path
        os.makedirs(out_path)

    vid_list = glob.glob(src_path+'/*/*.'+ext)
    print len(vid_list)

    #dump_frames("/home/zjg/zjg/tsncaffe/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi")

    for i in range(len(vid_list)):
        print "start processing: "+vid_list[i]
        dump_frames(vid_list[i], new_size)
    #for index , item in enumerate(vid_list):
    #    print "start processing: "+item
    #    dump_frames(item)
    print len(vid_list)

    #pool = Pool(num_worker)
    #if flow_type == 'tvl1':
    #    pool.map(run_optical_flow, zip(vid_list, xrange(len(vid_list))))
    #elif flow_type == 'warp_tvl1':
    #    pool.map(run_warp_optical_flow, zip(vid_list, xrange(len(vid_list))))