# -*- coding: utf-8 -*-
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np
import tensorflow as tf
mnist = input_data.read_data_sets("/media/kuang/D/dataset/mnist",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder(tf.float32,[None,10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#方式
for i in range(1000):
    b_x ,b_y = mnist.train.next_batch(100)
    feed_dict = {x:b_x,y_:b_y}
    sess.run(train_step,feed_dict=feed_dict)
correct_predection = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
s = tf.reduce_mean(tf.cast(correct_predection,tf.float32))

print(sess.run(s,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
 #大


