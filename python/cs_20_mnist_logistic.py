# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 05:29:13 2019
CS 20: TensorFlow for Deep Learning Research
Lecture 3
1/19/2017
"""
from tensorflow.examples.tutorials.mnist import input_data
import utils

MNIST = input_data.read_data_sets("G://My Drive//data//mnist//", one_hot=True)
mnist_folder = "G://My Drive//data//mnist//"
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)