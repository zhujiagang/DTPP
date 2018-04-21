
make clean
rm -rf build
mkdir build

#把之前编译的去掉

#export LD_LIBRARY_PATH=/home/user/cudnnv4/lib64:$LD_LIBRARY_PATH

make all -j16
#make test -j16

#make runtest -j16
make pycaffe -j16
#make matcaffe -j16
#1. 数据预处理
#sh data/mnist/get_mnist.sh
#2. 重建 lmdb 文件。Caffe 支持多种数据格式输入网络,包括 Image(.jpg, .png 等),leveldb,lmdb,HDF5 等,根据自己需要选择不同输入吧。
#sh examples/mnist/create_mnist.sh
#sh examples/mnist/train_lenet.sh
