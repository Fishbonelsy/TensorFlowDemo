from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ImageUtils


import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data")

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier(784)

# Train the model on some example data.
classifier.train(input_fn=mnist.train.next_batch(10000), steps=2000)

# Use it to predict.
image_data = ImageUtils.get_image_data()
predictions = classifier.predict(input_fn=image_data.reshape(1,784))
print (predictions)
