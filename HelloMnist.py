# coding=utf-8
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import ImageUtils
import numpy as np
from tensorflow.python.tools import freeze_graph

DATA_SIZE = 10000

FLAGS = None


def main(_):
    global extra_xs
    global extra_ys
    mnist = input_data.read_data_sets(FLAGS.data_dir)
    print(type(mnist))
    print(FLAGS.data_dir)
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784] , name='input')
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    temp = tf.matmul(x, W) + b
    y = tf.identity(temp, name='output')
    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])


    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
    # outputs of 'y', and then average across the batch.
    # 交叉熵
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    # 梯度下降算法
    train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print("Train")

    # # Steal data
    # batch_xs, batch_ys = mnist.train.next_batch(DATA_SIZE)
    # image_file = open("/Users/longsiyang/OceanProject/TensorFlow/image_data/image_data.csv", 'ab')
    # image_file.seek(0)
    # image_file.truncate()
    # np.savetxt(image_file, batch_xs)
    # np.savetxt("/Users/longsiyang/OceanProject/TensorFlow/image_data/label_data.csv", batch_ys)

    # Train
    for i in range(DATA_SIZE):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    print("Test")
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(
        accuracy, feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels
        }))

    # Use trained model
    image_data = ImageUtils.get_image_data()
    ret = sess.run(y, feed_dict={x: image_data.reshape(1, 784)})
    print(ret.argmax())

    # Save model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)  # 声明tf.train.Saver类用于保存模型
    tf.train.write_graph(sess.graph_def, "/Users/longsiyang/OceanProject/TensorFlow/model/","graph.pb", as_text=False)
    saver_path = saver.save(sess,
                            "/Users/longsiyang/OceanProject/TensorFlow/model/model.ckpt")  # 将模型参数保存到save/model.ckpt文件
    freeze_graph.freeze_graph("/Users/longsiyang/OceanProject/TensorFlow/model/graph.pb", '', True, saver_path, 'output', 'save/restore_all',
                              'save/Const:0', "/Users/longsiyang/OceanProject/TensorFlow/model/frozen_model.pb", False, "")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
