from utils import PluginFeatureExtractor
import tensorflow as tf
import numpy as np
from tqdm import trange
from tensorflow.python.ops import rnn, rnn_cell

# https://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/
# CNNs apply a series of filters to the raw pixel data of an image to extract
# and learn higher level features, which the model can then use for
# classification.


def conv_2d(x, weights, biases, strides=1):
    # Convolutional layers apply a spicifed number of convolution filters to the
    # image. For each subregion, the layer performs a set of mathematical ops
    # to produce a single value in the output feature map. Typically a ReLU
    # activation function is applied to introduce non-linearites to the model.

    # Stride is the stride of the sliding window for each dimension of input.
    # Padding is the type of padding algorithm used for pixels around the border.
    x = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')
    # Add bias to the inputs.
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool_2d(x, k=2):
    # Pooling layers downsample the image data extracted by convolutional layers
    # to reduce the dimensionality of the feature map in order to decrease
    # processing time. Max pooling is common, which extracts the maximimum value
    # from subregions of the feature map and discards all other values.

    # 'ksize' is a list of ints >= 4 that describes the size of the window for
    # each dimension of the input tensor. 'strides' is a list of ints again >= 4
    # that describes the stride of the sliding window for each dimension of the
    # tensor. Padding is the same as the conv2d description.
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):
    
