import tensorflow as tf
from tensorflow import keras
import math

kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)


def shape2d(a):
	"""
	Ensure a 2D shape.

	Args:
		a: a int or tuple/list of length 2

	Returns:
		list: of length 2. if ``a`` is a int, return ``[a, a]``.
	"""
	if type(a) == int:
		return [a, a]
	if isinstance(a, (list, tuple)):
		assert len(a) == 2
		return list(a)
	raise RuntimeError("Illegal shape: {}".format(a))


def ConvBlock(input_tensor, num_channels, kernel_size, stride, name, is_training=False, data_format='channels_last'):
	if data_format =='channel_first':
		axis = 1
	else:
		axis = -1
	x = keras.layers.Conv2D(num_channels, kernel_size=kernel_size, strides=stride, padding='same',
							use_bias=False, name='{}_conv'.format(name), data_format=data_format)(input_tensor)
	x = tf.layers.batch_normalization(x, axis=1, training=is_training)
	x = keras.layers.ReLU(name='{}_relu'.format(name))(x)
	return x


def MyDepthConv(x, kernel_shape, channel_mult=1, padding='SAME', stride=1, rate=1, data_format='NHWC',
				W_init=None, activation=tf.identity, name=None):
	in_shape = x.get_shape().as_list()
	if data_format == 'NHWC':
		in_channel = in_shape[3]
		stride_shape = [1, stride, stride, 1]
	elif data_format == 'NCHW':
		in_channel = in_shape[1]
		stride_shape = [1, 1, stride, stride]
	out_channel = in_channel * channel_mult

	if W_init is None:
		W_init = kernel_initializer
	kernel_shape = shape2d(kernel_shape)  # [kernel_shape, kernel_shape]
	filter_shape = kernel_shape + [in_channel, channel_mult]

	W = tf.get_variable(name, filter_shape, initializer=W_init)
	conv = tf.nn.depthwise_conv2d(x, W, stride_shape, padding=padding, rate=[rate, rate], data_format=data_format)

	return conv


def GhostModule(name, x, filters, kernel_size, dw_size, ratio, padding='SAME', strides=1,
				data_format='NHWC', use_bias=False, is_training=False, activation=True):
	if data_format == 'NCHW':
		axis = 1
	else:
		axis = -1

	with tf.variable_scope(name):
		init_channels = math.ceil(filters / ratio)
		if data_format == 'NCHW':
			x = tf.layers.Conv2D(init_channels, kernel_size, strides=strides, padding=padding,
							 	kernel_initializer=kernel_initializer, use_bias=use_bias, name=name + 'Conv', data_format='channels_first')(x)
		else:
			x = tf.layers.Conv2D(init_channels, kernel_size, strides=strides, padding=padding,
							 	kernel_initializer=kernel_initializer, use_bias=use_bias, name=name + 'Conv', data_format='channels_last')(x)
		x = tf.layers.batch_normalization(x, training=is_training, name=name+'BN_1', axis=axis)
		x = tf.nn.relu(x, name=name+'Relu_1')

		if ratio == 1:
			return x
		dw1 = MyDepthConv(x, [dw_size, dw_size], channel_mult=ratio - 1, stride=1, data_format=data_format, name=name)
		dw1 = tf.layers.batch_normalization(dw1, training=is_training, name=name+'BN_2', axis=axis)
		if activation:
			dw1 = tf.nn.relu(dw1, name=name + 'Relu_2')

		if data_format == 'NCHW':
			dw1 = dw1[:, :filters - init_channels, :, :]
			x = tf.concat([x, dw1], axis=axis)
		else:
			dw1 = dw1[:, :, :, :filters - init_channels]
			x = tf.concat([x, dw1], axis=axis)
		return x

