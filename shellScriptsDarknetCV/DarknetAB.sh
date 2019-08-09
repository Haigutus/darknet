#!/bin/bash

sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

git clone https://github.com/AlexeyAB/darknet.git DarknetAB

cd DarknetAB/

#sed -i 's/GPU=0/GPU=1/g' Makefile
sed -i '0,/GPU=0/s//GPU=1/' Makefile
sed -i '0,/CUDNN=0/s//CUDNN=1/' Makefile
sed -i '0,/OPENCV=0/s//OPENCV=1/' Makefile
sed -i '0,/LIBSO=0/s//LIBSO=1/' Makefile


make -j10

cd ../

git clone https://github.com/AlexeyAB/Yolo_mark.git

cd Yolo_mark/

cmake .
make -j10
chmod +x linux_mark.sh
#./linux_mark.sh
