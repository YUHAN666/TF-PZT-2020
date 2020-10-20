import tensorflow as tf
from config import ACTIVATION
from swish import swish
from mish import mish
from tensorflow.python.ops import math_ops

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)


def Squeeze_excitation_layer(input_x, out_dim, se_dim, name, data_format):
    with tf.name_scope(name):

        if data_format == 'channels_first':
            squeeze = math_ops.reduce_mean(input_x, [2, 3], name=name + '_gap', keepdims=True)
        else:
            squeeze = math_ops.reduce_mean(input_x, [1, 2], name=name + '_gap', keepdims=True)

        # excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=se_dim,
        #                              kernel_initializer=DENSE_KERNEL_INITIALIZER, name=name + '_fully_connected1')
        excitation = tf.layers.Conv2D(se_dim, (1, 1), strides=(1, 1), kernel_initializer=kernel_initializer, padding='same', data_format=data_format)(squeeze)
        if ACTIVATION == 'swish':
            excitation = swish(excitation, name=name + '_swish')
        elif ACTIVATION == 'mish':
            excitation = mish(excitation)
        else:
            excitation = tf.nn.relu(excitation, name=name + '_relu')

        # excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=out_dim,
        #                              kernel_initializer=DENSE_KERNEL_INITIALIZER, name=name + '_fully_connected2')
        excitation = tf.layers.Conv2D(out_dim, 1, strides=1, kernel_initializer=kernel_initializer, padding='same', data_format=data_format)(excitation)

        excitation = tf.nn.sigmoid(excitation, name=name + '_sigmoid')

        if data_format == 'channels_first':
            excitation = tf.reshape(excitation, [-1, out_dim, 1, 1])
        else:
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        scale = input_x * excitation

        return scale
