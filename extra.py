# 进行解算
for it in range(niter):
    solver.step(1)
        _train_loss += solver.net.blobs['rgb_flow_gating_loss'].data
        if it % display == 0:
            train_loss[it // display] = _train_loss / display
            _train_loss = 0

        if it % test_interval == 0:
            print '\n my test, train iteration', it
            for test_it in range(test_iter):
                #print '\n my test, test iteration \n', test_it
                solver.test_nets[0].forward()
                _test_loss += solver.test_nets[0].blobs['rgb_flow_gating_loss'].data
                _accuracy += solver.test_nets[0].blobs['rgb_flow_gating_accuracy'].data
            test_loss[it / test_interval] = _test_loss / test_iter
            test_acc[it / test_interval] = _accuracy / test_iter
            if _max_accuracy < test_acc[it / test_interval]:
                _max_accuracy = test_acc[it / test_interval]
                _max_accuracy_iter = it
                solver.net.save('/home/zhujiagang/temporal-segment-networks/models/ucf101_split_1_gating_three_iter_' + str(it) + '.caffemodel')
                print '\nnewly max: _max_accuracy and _max_accuracy_iter', _max_accuracy, _max_accuracy_iter
            print '\n_max_accuracy and _max_accuracy_iter', _max_accuracy, _max_accuracy_iter
            _test_loss = 0
            _accuracy = 0

    print '\nplot the train loss and test accuracy\n'
    print '\n_max_accuracy and _max_accuracy_iter\n', _max_accuracy, _max_accuracy_iter

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # train loss -> 绿色
    ax1.plot(display * arange(len(train_loss)), train_loss, 'g')
    # test loss -> 黄色
    ax1.plot(test_interval * arange(len(test_loss)), test_loss, 'y')
    # test accuracy -> 红色
    ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    plt.show()