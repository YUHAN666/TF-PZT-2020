import tensorflow as tf
import numpy as np
from swish import swish
from mish import mish


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """
    Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def ConvBatchNormRelu(inputs, filters, kernel_size, strides, training, momentum, mode, name=None, padding='same',
        data_format='channels_last', activation=None, bn=True, use_bias=False):

    axis = 1 if data_format == 'channels_first' else -1

    if mode != "savePb":  # BN without bias
        x = tf.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, data_format=data_format,
                             name=name + '_CBR_Conv2D', padding=padding, use_bias=use_bias,
                             kernel_initializer=conv_kernel_initializer)(inputs)
        if bn:
            x = tf.layers.batch_normalization(x, axis=axis, training=training, momentum=momentum, name=name + '_CBR_bn')

    else:  # bias without BN
        x = tf.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, data_format=data_format,
                             name=name + '_CBR_Conv2D', padding=padding, use_bias=True,
                             kernel_initializer=conv_kernel_initializer)(inputs)

    if activation == 'relu':
        x = tf.nn.relu(x, name=name + '_CBR_relu')
    elif activation == 'swish':
        x = swish(x, name=name+'_CBR_swish')
    elif activation == 'mish':
        x = mish(x, name=name+'_CBR_mish')
    else:
        pass

    return x