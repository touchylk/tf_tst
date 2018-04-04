from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import tensorflow as tf
mnist = input_data.read_data_sets("/home/kuang/dataset/",one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

cv2.imshow('a',(mnist.train.images[133]).reshape([28,28]))
cv2.waitKey(0)