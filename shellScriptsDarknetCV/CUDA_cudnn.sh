#!/bin/bash

sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

# install CUDA Toolkit v9.0
# instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb)
CUDA_REPO_PKG="cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb"
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/${CUDA_REPO_PKG}
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-9-0

CUDA_PATCH1="cuda-repo-ubuntu1604-9-0-local-cublas-performance-update_1.0-1_amd64-deb"
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/1/${CUDA_PATCH1}
sudo dpkg -i ${CUDA_PATCH1}
sudo apt-get update

# install cuDNN v7.0
#CUDNN_PKG="libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb"
#wget https://github.com/ashokpant/cudnn_archive/raw/master/v7.0/${CUDNN_PKG}
#sudo dpkg -i ${CUDNN_cdPKG}
#sudo apt-get update
git clone https://github.com/martinsimon88/CV-setup.git
cd CV-setup/
cat part* > cudnn-9.0-linux-x64-v7.4.1.5.tgz
tar -xzvf cudnn-9.0-linux-x64-v7.4.1.5.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda-9.0/lib64/libcudnn*

# install NVIDIA CUDA Profile Tools Interface ( libcupti-dev v9.0)
sudo apt-get install cuda-command-line-tools-9-0

# set environment variables
echo 'export PATH=${PATH}:/usr/local/cuda-9.0/bin' >> ~/.bashrc 
echo 'export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64' >> ~/.bashrc 
source ~/.bashrc
reboot
