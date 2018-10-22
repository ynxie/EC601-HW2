import tensorflow as tf
import numpy as np


def conv_2d(inputs, filters, kernel_size, name=None):
  """3x3 conv layer: ReLU + (1, 1) stride + He initialization"""

  # He initialization = normal dist with stdev = sqrt(2.0/fan-in)
  stddev = np.sqrt(2 / (np.prod(kernel_size) * int(inputs.shape[3])))
  out = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                         padding='same', activation=tf.nn.relu,
                         kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                         name=name)
  tf.summary.histogram('act' + name, out)

  return out


def dense_relu(inputs, units, name=None):
  """3x3 conv layer: ReLU + He initialization"""

  # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
  stddev = np.sqrt(2 / int(inputs.shape[1]))
  out = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                        kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                        name=name)

  tf.summary.histogram('act' + name, out)

  return out


def dense(inputs, units, name=None):
  """3x3 conv layer: ReLU + He initialization"""

  # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
  stddev = np.sqrt(2 / int(inputs.shape[1]))
  out = tf.layers.dense(inputs, units,
                        kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                        name=name)
  tf.summary.histogram('act' + name, out)

  return out


def cnn(training_batch, config):
  """VGG-like conv-net

  Args:
    training_batch: batch of images (N, 56, 56, 3)
    config: training configuration object

  Returns:
    class prediction scores
  """
  img = tf.cast(training_batch, tf.float32)
  out = (img - 128.0) / 128.0

  tf.summary.histogram('img', training_batch)
  # (N, 56, 56, 3)
  out = conv_2d(out, 64, (3, 3), 'conv1')
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool1')

  # (N, 28, 28, 64)
  out = conv_2d(out, 128, (3, 3), 'conv2')
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool2')

  # (N, 14, 14, 128)

  # fc1: flatten -> fully connected layer
  # (N, 14, 14, 128)-> (N, 25088) -> (N, 2048)
  out = tf.contrib.layers.flatten(out)
  out = dense_relu(out, 2048, 'fc1')
  out = tf.nn.dropout(out, config.dropout_keep_prob)


  # softmax
  # (N, 2048) -> (N, 5)
  logits = dense(out, config.class_num, 'fc2')

  return logits
