#!/bin/bash

sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

sudo apt install python-dev python-pip
sudo pip install -U virtualenv  # system-wide install

virtualenv --system-site-packages -p python2.7 ./venv

source ./venv/bin/activate  # sh, bash, ksh, or zsh

pip install --upgrade pip
pip install --upgrade tensorflow-gpu

python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
