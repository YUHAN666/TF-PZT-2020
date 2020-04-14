import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import re

netconfig = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
			 [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
			 [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]


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


def shuffle_channel(x, groups):
	with tf.variable_scope('shuffle_unit'):
		n, h, w, c = x.get_shape().as_list()
		x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
		x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
		x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
		return x


def swish_activation(x, name=None):
	# return tf.nn.relu(x,name = name)
	with ops.name_scope(name, "my_swish", [x]) as name:

		"""Swish activation function: x * sigmoid(x).
		Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
		"""
		try:
			# The native TF implementation has a more
			# memory-efficient gradient implementation
			return tf.nn.swish(x)
		except AttributeError:
			pass

		return x * tf.nn.sigmoid(x)


def Squeeze_excitation_layer(input_x, out_dim, se_dim, swish=False, name='SE'):
	with tf.name_scope(name):

		squeeze = math_ops.reduce_mean(input_x, [1, 2], name=name + '_gap', keepdims=True)

		excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=se_dim, name=name + '_fully_connected1')

		if swish:
			excitation = swish_activation(excitation, name=name + '_swish')
		else:
			excitation = tf.nn.relu(excitation, name=name + '_relu')

		excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=out_dim, name=name + '_fully_connected2')

		excitation = tf.nn.sigmoid(excitation, name=name + '_sigmoid')

		excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

		scale = input_x * excitation

		return scale


def S2block(inputs, nOut, netconfig, training, bn_momentum, name):
	kSize = netconfig[0]
	avgsize = netconfig[1]
	if avgsize > 1:
		tmp = tf.layers.average_pooling2d(inputs, pool_size=(avgsize, avgsize), strides=(avgsize, avgsize),
										  name=name + '_agp2d')
	else:
		tmp = inputs

	x = tf.keras.layers.DepthwiseConv2D(kSize, 1, depth_multiplier=1, padding='same')(tmp)
	x = tf.layers.batch_normalization(x, training=training, momentum=bn_momentum, name=name + '_bn1')
	x = tf.nn.relu(x, name=name + '_relu')
	x = tf.layers.conv2d(x, nOut, 1, 1, name=name + '_conv2d', padding='same', use_bias=False)

	if avgsize > 1:
		x = tf.keras.layers.UpSampling2D((avgsize, avgsize))(x)

	x = tf.layers.batch_normalization(x, training=training, momentum=bn_momentum, name=name + '_bn2')

	return x


def MDConv(inputs, kernel, strides, kernel_initializer, padding='same', use_bias=False, name=None):
	"""
	grouped depthwise_conv. kernal size can be different for each group
	"""
	channel_axis = -1
	input_channles = inputs.get_shape().as_list()[channel_axis]

	if isinstance(kernel, list) or isinstance(kernel, tuple):
		groups = len(kernel)
		split_inputs = split_channels(input_channles, groups)
		x_splits = tf.split(inputs, split_inputs, channel_axis)
		out = []
		for n in range(groups):
			y = tf.keras.layers.DepthwiseConv2D(kernel[n], strides, depth_multiplier=1, padding=padding,
												use_bias=use_bias,
												kernel_initializer=kernel_initializer)(x_splits[n])
			out.append(y)
		out = tf.concat(out, axis=channel_axis)
	else:
		out = tf.keras.layers.DepthwiseConv2D(kernel, strides, depth_multiplier=1, padding=padding, use_bias=use_bias,
											  kernel_initializer=kernel_initializer)(inputs)
	return out


def split_channels(total_filters, num_groups):
	split = [total_filters // num_groups for _ in range(num_groups)]
	split[0] += total_filters - sum(split)
	return split


def GroupedConv2D(inputs, filters, kernel, strides, kernel_initializer, padding='same', use_bias=False, name=None):
	channel_axis = -1
	input_channles = inputs.get_shape().as_list()[channel_axis]

	if isinstance(kernel, list) or isinstance(kernel, tuple):
		groups = len(kernel)
		split_inputs = split_channels(input_channles, groups)
		split_filters = split_channels(filters, groups)
		x_splits = tf.split(inputs, split_inputs, channel_axis)
		out = []
		for n in range(groups):
			y = tf.layers.conv2d(x_splits[n], split_filters[n], kernel_size=kernel[n], strides=strides,
								 name=name + '_GroupedConv' + str(n),
								 padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer)
			out.append(y)
		out = tf.concat(out, axis=channel_axis)
	else:
		out = tf.layers.conv2d(inputs, filters, kernel_size=kernel, strides=strides, name=name + '_Conv2D',
							   padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer)
	return out


def SEseparableCBR(inputs, nOut, kernel, t, s, training, bn_momentum, se_ratio, name=None, SE=True):
	nIn = inputs.get_shape().as_list()[-1]
	tchannel = nIn * t

	# expand different from original SInet,which is without expand
	# x = tf.layers.conv2d(inputs, tchannel, kernel_size=(1, 1), strides=(1, 1), name=name + 'expand_Conv2D',
	# 					 padding='same', use_bias=False)

	# x = tf.layers.batch_normalization(x, training=training, momentum=bn_momentum, name=name + 'expand_bn')
	# x = tf.nn.relu(x, name=name + 'expand_relu')

	# DepthwiseConv
	x = tf.keras.layers.DepthwiseConv2D(kernel, s, depth_multiplier=1, padding='same')(inputs)
	x = tf.layers.batch_normalization(x, training=training, momentum=bn_momentum, name=name + '_dsconv_bn')
	x = tf.nn.relu(x, name=name + 'dsconv_relu')

	# SE
	if SE:
		num_reduced_filters = max(1, int(nIn * se_ratio))
		x = Squeeze_excitation_layer(x, tchannel, num_reduced_filters, swish=False, name=name + 'se_layer')
	# project conv
	x = tf.layers.conv2d(x, nOut, kernel_size=(1, 1), strides=(1, 1), name=name + 'project_Conv2D',
						 padding='same', use_bias=False)

	x = tf.layers.batch_normalization(x, training=training, momentum=bn_momentum, name=name + 'project_bn')
	x = tf.nn.relu(x, name=name + 'projct_relu')

	return x


def S2module(inputs, nOut, identity_map=True, netconfig=[[3, 1], [5, 1]], training=False, bn_momentum=0.99,
			 name='S2module'):
	group_n = len(netconfig)
	n = nOut // group_n
	n1 = nOut - group_n * n
	kernal = [1] * group_n

	x = GroupedConv2D(inputs, n, kernal, strides=(1, 1),
					  kernel_initializer=conv_kernel_initializer, name=name + 'GroupConv')

	x = shuffle_channel(x, group_n)

	y = []
	for i in range(group_n):
		if i == 0:
			y.append(S2block(x, n + n1, netconfig[i], training=training, bn_momentum=bn_momentum,
							 name=name + '_0' + '_group'))
		else:
			y.append(S2block(x, n, netconfig[i], training=training, bn_momentum=bn_momentum,
							 name=name + '_' + str(i) + '_group'))

	x = tf.concat(y, axis=3)

	if identity_map:
		x = x + inputs

	x = tf.layers.batch_normalization(x, training=training, momentum=bn_momentum, name=name + '_bn')
	x = tf.nn.relu(x, name=name + '_relu')

	return x


def SINet(inputs, classes=20, p=5, q=3, chnn=1, training=True, bn_momentum=0.99, dropout_rate=0.2, scope='SINet',
		  reuse=True):
	with tf.variable_scope(scope, reuse=reuse):

		dim1 = 16
		dim2 = 48 + 4 * (chnn - 1)
		dim3 = 96 + 4 * (chnn - 1)

		output1 = tf.layers.conv2d(inputs, 12, 3, 2, name='conv2d', padding='same', use_bias=False)
		output1 = tf.layers.batch_normalization(output1, training=training, momentum=bn_momentum, name='l1_bn')
		output1 = tf.nn.relu(output1, name='l1_relu')

		output2_0 = SEseparableCBR(output1, dim1, 3, 6, 2, training=training, bn_momentum=bn_momentum,
								   se_ratio=0.5, name='l2_SE_ds_CBR', SE=False)

		for i in range(0, p):
			if i == 0:  # inputs, nOut, identity_map=True, netconfig= [[3,1],[5,1]],training,bn_momentum,name
				output2 = S2module(output2_0, dim2, identity_map=False, netconfig=netconfig[i],
								   training=training, bn_momentum=bn_momentum, name='l2_S2module_' + str(i))
			else:
				output2 = S2module(output2, dim2, identity_map=True, netconfig=netconfig[i],
								   training=training, bn_momentum=bn_momentum, name='l2_S2module_' + str(i))

		output3_0 = tf.concat([output2_0, output2], axis=3)
		output3_0 = tf.layers.batch_normalization(output3_0, training=training, momentum=bn_momentum, name='l3_bn')
		output3_0 = tf.nn.relu(output3_0, name='l3_relu')

		# SEseparableCBR(dim2+dim1,dim2, 3,2, divide=2)
		output3_0 = SEseparableCBR(output3_0, dim2, 3, 6, 2, training=training, bn_momentum=bn_momentum,
								   se_ratio=0.5, name='level3_SE_ds_CBR', SE=False)

		for i in range(0, q):
			if i == 0:  # inputs, nOut, identity_map=True, netconfig= [[3,1],[5,1]],training,bn_momentum,name
				output3 = S2module(output3_0, dim3, identity_map=False, netconfig=netconfig[i + p],
								   training=training, bn_momentum=bn_momentum, name='l3_S2module_' + str(i))
			else:
				output3 = S2module(output3, dim3, identity_map=True, netconfig=netconfig[i + p],
								   training=training, bn_momentum=bn_momentum, name='l3_S2module_' + str(i))

		output3_cat = tf.concat([output3_0, output3], axis=3)
		output3_cat = tf.layers.batch_normalization(output3_cat, training=training, momentum=bn_momentum,
													name='l3cat_bn')
		output3_cat = tf.nn.relu(output3_cat, name='l3cat_relu')

		Enc_final = tf.layers.conv2d(output3_cat, dim2, 1, 1, name='enc_final_conv', padding='same', use_bias=False)

		up_lebel2 = tf.keras.layers.UpSampling2D((2, 2))(Enc_final)
		up_lebel2 = tf.layers.batch_normalization(up_lebel2, training=training, momentum=bn_momentum, name='up2_bn')

		confidence = tf.layers.conv2d(up_lebel2, classes, 3, 1, name='confidence_conv', padding='same', use_bias=True)
		confidence = tf.nn.sigmoid(confidence, name='confidence_sigmoid')
		# confidence = tf.reduce_max(confidence,axis=-1,keepdims=True,name='confiddence_max')
		gate = 1 - confidence
		gate = tf.tile(gate, [1, 1, 1, dim2])

		out2_skip = tf.layers.conv2d(output2, dim2, 3, 1, name='skip_conv2d', padding='same', use_bias=False)
		out2_skip = tf.layers.batch_normalization(out2_skip, training=training, momentum=bn_momentum, name='skip_bn')
		out2_skip = tf.nn.relu(out2_skip, name='skip_relu')

		up_level1 = out2_skip * gate + up_lebel2
		# up_level1 = tf.keras.layers.UpSampling2D((2, 2))(up_level1)
		# up_level1 = tf.layers.batch_normalization(up_level1, training=training, momentum=bn_momentum, name='up1_bn')

		feature0 = tf.keras.layers.UpSampling2D((2, 2))(up_level1)
		logits = tf.layers.conv2d(feature0, classes, 3, 1, name='logits_conv', padding='same', use_bias=True)

		return [feature0, logits]