# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:22:22 2016

@author: root
"""
from multiprocessing import Process

import numpy as np
from numpy import zeros,arange
import matplotlib.pyplot as plt
import sys
import os
from math import ceil
import cv2
from matplotlib.pyplot import twinx
from math import ceil

caffe_root = '/home/zhujiagang/temporal-segment-networks/lib/caffeaction/'
sys.path.insert(0, caffe_root + 'python')
#sys.path.insert(0, caffe_root + 'python/train_2.py')
import caffe
#from train_2 import train_this


def train(
        solver,  # solver proto definition
        snapshot,  # solver snapshot to restore
        gpus,  # list of device ids
        timing=False,  # show timing info for compute and communications
):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve_step,
                    args=(solver, snapshot, gpus, timing, uid, rank))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))


def solve(proto, snapshot, gpus, timing, uid, rank):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)
    if snapshot and len(snapshot) != 0:
        solver.restore(snapshot)

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        time(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    solver.step(solver.param.max_iter)

def solve_step(proto, snapshot, gpus, timing, uid, rank):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)
    if snapshot and len(snapshot) != 0:
        solver.restore(snapshot)

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        time(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    niter = solver.param.max_iter
    display = solver.param.display
    test_iter = 950
    test_interval = 200
    # 初始化
    train_loss = zeros(int(ceil(niter // display)))
    test_loss = zeros(int(ceil(niter // test_interval)))
    test_acc = zeros(int(ceil(niter // test_interval)))
    # 辅助变量
    _train_loss = 0;
    _test_loss = 0;
    _accuracy = 0;
    _max_accuracy = 0;
    _max_accuracy_iter = 0;
    # 进行解算
    for it in range(niter):
        solver.step(1)


#def train_this(
#        solver,  # solver proto definition
#        snapshot,  # solver snapshot to restore
#        gpus,  # list of device ids
#        timing=False,  # show timing info for compute and communications
#):
 #   train(solver, snapshot, gpus, timing)

#solver = caffe.SGDSolver('/home/zhujiagang/temporal-segment-networks/models/ucf101/gating_three_solver.prototxt')
#solver.restore('/home/zhujiagang/temporal-segment-networks/models/ucf101_split_1_gating_three_iter_200.solverstate')
train(solver = '/home/zhujiagang/temporal-segment-networks/models/ucf101/gating_three_solver.prototxt',
           snapshot = '/home/zhujiagang/temporal-segment-networks/models/ucf101_split_1_gating_three_iter_200.solverstate',
           gpus = [0,1],
           timing = False)