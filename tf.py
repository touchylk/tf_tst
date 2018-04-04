from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np
import tensorflow as tf
mnist = input_data.read_data_sets("/home/kuang/dataset/",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder(tf.float32,[None,10])
loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    b_x ,b_y = mnist.train.next_batch(100)
    feed_dict = {x:b_x,y_:b_y}
    train_step.run(feed_dict)
y2= tf.nn.softmax(tf.matmul((mnist.train.images[13432]).reshape([1,784]), w)+b)



print(sess.run(y2))
cv2.imshow('a',(mnist.train.images[13432]).reshape([28,28]))
cv2.waitKey(0)