""" Implementation of modules use in GhostNet.py """
import tensorflow as tf
from tensorflow import keras
import math
from swish import swish
from mish import mish
from convblock import ConvBatchNormRelu as CBR

kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
# def ConvBlock(input_tensor, num_channels, kernel_size, stride, name, is_training=False, data_format='channels_last', activation='relu'):
# 	if data_format =='channel_first':
# 		axis = 1
# 	else:
# 		axis = -1
# 	x = keras.layers.Conv2D(num_channels, kernel_size=kernel_size, strides=stride, padding='same',
# 							use_bias=False, name='{}_conv'.format(name), data_format=data_format)(input_tensor)
# 	x = tf.layers.batch_normalization(x, training=is_training)
# 	if activation == 'relu':
# 		x = keras.layers.ReLU(name='{}_relu'.format(name))(x)
# 	elif activation == 'mish':
# 		x = mish(x)
# 	elif activation =='swish':
# 		x = swish(x, name+'_swish')
# 	return x


def MyDepthConv(x, kernel_shape, channel_mult=1, padding='SAME', stride=1, rate=1, data_format='channels_first',
				W_init=None, activation=tf.identity, name=None):
	in_shape = x.get_shape().as_list()

	if data_format == 'channels_first':
		in_channel = in_shape[1]
		stride_shape = [1, 1, stride, stride]
		data_format = 'NCHW'
	else:
		in_channel = in_shape[-1]
		stride_shape = [1, stride, stride, 1]
		data_format = 'NHWC'

	out_channel = in_channel * channel_mult

	if W_init is None:
		W_init = kernel_initializer
	filter_shape = kernel_shape + [in_channel, channel_mult]

	W = tf.get_variable(name+"_weight", filter_shape, initializer=W_init)
	conv = tf.nn.depthwise_conv2d(x, W, stride_shape, padding=padding, rate=[rate, rate], data_format=data_format, name=name+"_depthwise_conv2d")

	return conv


def GhostModule(name, x, filters, kernel_size, dw_size, ratio, mode, padding='SAME', strides=1,
				data_format='channels_first', use_bias=False, is_training=False, activation='relu', momentum=0.9):
	if data_format == 'channels_first':
		axis = 1
	else:
		axis = -1

	with tf.variable_scope(name):
		init_channels = math.ceil(filters / ratio)

		x = CBR(x, init_channels, kernel_size, strides=strides, training=is_training, momentum=momentum, mode=mode,
				name=name, padding='same', data_format=data_format, activation='relu', bn=True, use_bias=use_bias)
		# x = tf.layers.Conv2D(init_channels, kernel_size, strides=strides, padding=padding,
		# 					kernel_initializer=kernel_initializer, use_bias=use_bias, name=name + 'Conv', data_format='channels_last')(x)
		# x = tf.layers.batch_normalization(x, training=is_training, name=name+'BN_1')
		# if activation =='relu':
		# 	x = tf.nn.relu(x, name=name+'Relu_1')
		# elif activation == 'mish':
		# 	x = mish(x)
		# elif activation == 'swish':
		# 	x = swish(x, name=name+'swish_1')
		# else:
		# 	pass

		if ratio == 1:
			return x
		dw1 = MyDepthConv(x, [dw_size, dw_size], channel_mult=ratio - 1, stride=1, data_format=data_format, name=name)
		dw1 = tf.layers.batch_normalization(dw1, training=is_training, name=name+'BN_2', axis=axis)
		if activation == 'relu':
			dw1 = tf.nn.relu(dw1, name=name + 'Relu_2')
		elif activation == 'mish':
			dw1 = mish(dw1)
		elif activation == 'swish':
			dw1 = swish(dw1, name=name+'swish_2')
		else:
			pass

		if data_format == 'channels_first':
			dw1 = dw1[:, :filters - init_channels, :, :]
		else:
			dw1 = dw1[:, :, :, :filters - init_channels]
		x = tf.concat([x, dw1], axis=axis)
		return x

