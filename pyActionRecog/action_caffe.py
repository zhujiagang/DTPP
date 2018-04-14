import sys
proto_root = "/home2/lin_li/anaconda2/pkgs/libprotobuf-3.2.0-0/lib/"
sys.path.insert(0, proto_root)

import caffe
from caffe.io import oversample
import numpy as np
from utils.io import flow_stack_oversample, fast_list2arr, rgb_stack_oversample, oversample_for_rgb_stack, flow_stack_oversample_new, oversample_for_flow_stack_test
import cv2
import matplotlib.pyplot as plt


class CaffeNet(object):

    def __init__(self, net_proto, net_weights, device_id, input_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        self._net = caffe.Net(net_proto, net_weights, caffe.TEST)

        input_shape = self._net.blobs['data'].data.shape

        if input_size is not None:
            input_shape = input_shape[:2] + input_size

        transformer = caffe.io.Transformer({'data': input_shape})

        #if self._net.blobs['data'].data.shape[1] == 3:
            #printf
         #   transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
         #   transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        #else:
        #    pass # non RGB data need not use transformer

        self._transformer = transformer

        self._sample_shape = self._net.blobs['data'].data.shape

    def predict_single_frame(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None):

        if frame_size is not None:
            frame1 = fast_list2arr([x for x in frame])
            frame = [cv2.resize(x, frame_size) for x in frame]

        #print frame1.shape

        if over_sample:
            if multiscale is None:
                os_frame = oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)

        #print os_frame.shape

        #data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])
        def preprocess_1(r):
            r = r.transpose(2,0,1)
            r[0,:,:] = r[0,:,:] - 104
            r[1,:,:] = r[1,:,:] - 117
            r[2,:,:] = r[2,:,:] - 123
            return r

        data = fast_list2arr([preprocess_1(x) for x in os_frame])

        #print data.shape


        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()


    def predict_single_rgb_stack(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None, stack_len=25):

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]

        if over_sample:
            if multiscale is None:
                os_frame = oversample_for_rgb_stack(frame, (self._sample_shape[2], self._sample_shape[3]),stack_len)
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)

        def preprocess_1(r):
            r = r.transpose(2,0,1)
            r[0,:,:] = r[0,:,:] - 104
            r[1,:,:] = r[1,:,:] - 117
            r[2,:,:] = r[2,:,:] - 123
            return r

        data = fast_list2arr([preprocess_1(x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()

        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_rgb_stack_memory(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None, stack_len=25):

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]

        if over_sample:
            if multiscale is None:
                os_frame = oversample_for_rgb_stack(frame, (self._sample_shape[2], self._sample_shape[3]),stack_len)
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)

        def preprocess_1(r):
            r = r.transpose(2,0,1)
            r[0,:,:] = r[0,:,:] - 104
            r[1,:,:] = r[1,:,:] - 117
            r[2,:,:] = r[2,:,:] - 123
            return r

        data = fast_list2arr([preprocess_1(x) for x in os_frame])

        # self._net.blobs['data'].reshape(*data.shape)
        # self._net.reshape()
        #
        # out = self._net.forward(blobs=[score_name,], data=data)
        # return out[score_name].copy()

        data_new = data.reshape(-1,3*stack_len,224,224)
        scores_new = []
        for i in range(10):
            data_ele = data_new[i]
            self._net.blobs['data'].reshape(*data_ele.shape)
            self._net.reshape()
            out = self._net.forward(blobs=[score_name,], data=data_ele)
            scores_new.append(out[score_name].copy())
        scores_new = np.array(scores_new).reshape(10,101)
        return scores_new


    def predict_single_flow_stack_test(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None, stack_len=25):

        if over_sample:
            if multiscale is None:
                os_frame = oversample_for_flow_stack_test(frame, (self._sample_shape[2], self._sample_shape[3]),stack_len)
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)

        os_frame = np.array(os_frame).transpose(0,3,1,2)
        data = os_frame  - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_flow_stack_test_memory(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None, stack_len=25):

        if over_sample:
            if multiscale is None:
                os_frame = oversample_for_flow_stack_test(frame, (self._sample_shape[2], self._sample_shape[3]),stack_len)
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)

        os_frame = np.array(os_frame).transpose(0,3,1,2)
        data = os_frame  - np.float32(128.0)

        # self._net.blobs['data'].reshape(*data.shape)
        # self._net.reshape()
        # out = self._net.forward(blobs=[score_name,], data=data)
        # return out[score_name].copy()

        data_new = data.reshape(-1,10*stack_len,224,224)
        scores_new = []
        for i in range(10):
            data_ele = data_new[i]
            self._net.blobs['data'].reshape(*data_ele.shape)
            self._net.reshape()
            out = self._net.forward(blobs=[score_name,], data=data_ele)
            scores_new.append(out[score_name].copy())
        scores_new = np.array(scores_new).reshape(10,101)
        return scores_new

    def predict_single_flow_stack(self, frame, score_name, over_sample=True, frame_size=None):

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size) for x in frame])
        else:
            frame = fast_list2arr(frame)

        if over_sample:
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_flow_stack_feature_map(self, frame, score_name, over_sample=False, frame_size=None, blobname = 'conv1/7x7_s2', dim = 30):

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size) for x in frame])
        else:
            frame = fast_list2arr(frame)

        print "frame", frame.shape

        if over_sample:
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        print "os_frame", os_frame.shape
    # (10, 256, 340)
    # (10, 10, 224, 224)

        data = os_frame - np.float32(128.0)
        print data.shape
        #self._net.blobs['data'].reshape(*data.shape)
        print self._net.blobs['data'].data[0].shape
        self._net.blobs['data'].data[...] = data
        #self._net.reshape()
        out = self._net.forward()#data=data

        feat = self._net.blobs[blobname].data[0,:dim]

        return feat.copy()




    def predict_single_flow_rgb_stack(self, flow_frame, rgb_frame, score_name, over_sample=True, frame_size=None, multiscale=None, score_name_1=None):

        flow_1 = fast_list2arr([cv2.resize(x, frame_size) for x in flow_frame])
        flow_2 = flow_stack_oversample(flow_1, (self._sample_shape[2], self._sample_shape[3]))
        flow_data = flow_2 - np.float32(128.0)

        rgb_1 = [cv2.resize(x, frame_size) for x in rgb_frame]
        rgb_2 = oversample(rgb_1, (self._sample_shape[2], self._sample_shape[3]))

        #rgb_data1 = fast_list2arr(os_frame_rgb)
       # print rgb_data1.shape

        def preprocess_1(r):
            r = r.transpose(2,0,1)
            r[0,:,:] = r[0,:,:] - 104
            r[1,:,:] = r[1,:,:] - 117
            r[2,:,:] = r[2,:,:] - 123
            return r

        rgb_data = fast_list2arr([preprocess_1(x) for x in rgb_2])

        #print flow_data.shape
        #print rgb_data.shape
        #flow_data = np.reshape(flow_data, (10,-1,224,224))
        rgb_data = np.reshape(rgb_data, (10,-1,224,224))
        #print flow_data.shape
        #print rgb_data.shape

        #data = np.array([], dtype = rgb_data[0].dtype)
        data = np.concatenate((flow_data, rgb_data), axis=1)

        #print data.shape

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        if score_name_1 is not None:
            out_1 = self._net.forward(blobs=[score_name_1,], data=data)
            return out[score_name].copy(), out_1[score_name_1].copy()

        return out[score_name].copy()

    def predict_single_flow_rgb_stack_3(self, flow_frame, rgb_frame, score_name, over_sample=True, frame_size=None,
                                      multiscale=None, score_name_1=None, score_name_2=None):

        flow_1 = fast_list2arr([cv2.resize(x, frame_size) for x in flow_frame])
        flow_2 = flow_stack_oversample(flow_1, (self._sample_shape[2], self._sample_shape[3]))
        flow_data = flow_2 - np.float32(128.0)

        rgb_1 = [cv2.resize(x, frame_size) for x in rgb_frame]
        rgb_2 = oversample(rgb_1, (self._sample_shape[2], self._sample_shape[3]))

        # rgb_data1 = fast_list2arr(os_frame_rgb)
        # print rgb_data1.shape

        def preprocess_1(r):
            r = r.transpose(2, 0, 1)
            r[0, :, :] = r[0, :, :] - 104
            r[1, :, :] = r[1, :, :] - 117
            r[2, :, :] = r[2, :, :] - 123
            return r

        rgb_data = fast_list2arr([preprocess_1(x) for x in rgb_2])

        # print flow_data.shape
        # print rgb_data.shape
        # flow_data = np.reshape(flow_data, (10,-1,224,224))
        rgb_data = np.reshape(rgb_data, (10, -1, 224, 224))
        # print flow_data.shape
        # print rgb_data.shape

        # data = np.array([], dtype = rgb_data[0].dtype)
        data = np.concatenate((flow_data, rgb_data), axis=1)

        # print data.shape

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name, ], data=data)
        if score_name_1 is not None and score_name_2 is not None:
            out_1 = self._net.forward(blobs=[score_name_1, ], data=data)
            out_2 = self._net.forward(blobs=[score_name_2, ], data=data)
            # print "here"
            return out[score_name].copy(), out_1[score_name_1].copy(), out_2[score_name_2].copy()

        return out[score_name].copy()