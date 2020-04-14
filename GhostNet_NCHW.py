from tensorflow import keras
from collections import namedtuple
import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
import cv2
from timeit import default_timer as timer
import math
from tensorflow.python.ops import math_ops

from config import IMAGE_SIZE

"------------------------------------Model  	Component------------------------------------------------"
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'factor', 'se'])
Bottleneck = namedtuple('Bottleneck', ['kernel', 'stride', 'depth', 'factor', 'se'])

_CONV_DEFS_0 = [
	Conv(kernel=[3, 3], stride=2, depth=16, factor=1, se=0),
	Bottleneck(kernel=[3, 3], stride=1, depth=16, factor=1, se=0),

	Bottleneck(kernel=[3, 3], stride=2, depth=24, factor=48 / 16, se=0),
	Bottleneck(kernel=[3, 3], stride=1, depth=24, factor=72 / 24, se=0),

	Bottleneck(kernel=[5, 5], stride=2, depth=40, factor=72 / 24, se=1),
	Bottleneck(kernel=[5, 5], stride=1, depth=40, factor=120 / 40, se=1),

	Bottleneck(kernel=[3, 3], stride=2, depth=80, factor=240 / 40, se=0),
	Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=200 / 80, se=0),
	Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184 / 80, se=0),
	Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184 / 80, se=0),

	Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=480 / 80, se=1),
	Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=672 / 112, se=1),
	Bottleneck(kernel=[5, 5], stride=2, depth=160, factor=672 / 112, se=1),

	Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=0),
	Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=1),
	Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=0),
	Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=1),

	Conv(kernel=[1, 1], stride=1, depth=960, factor=1, se=0),
	Conv(kernel=[1, 1], stride=1, depth=1280, factor=1, se=0)
]


def shape2d(a):

	if type(a) == int:
		return [a, a]
	if isinstance(a, (list, tuple)):
		assert len(a) == 2
		return list(a)
	raise RuntimeError("Illegal shape: {}".format(a))


def ConvBlock(input_tensor, num_channels, kernel_size, stride, name, is_training=False, data_format='channels_first'):
	if data_format =='channel_first':
		axis = 1
	else:
		axis = -1
	x =tf.layers.conv2d(input_tensor, num_channels, kernel_size=kernel_size, strides=stride, padding='same',
									use_bias=False, name='{}_conv'.format(name), data_format=data_format)
	x = tf.layers.batch_normalization(x, axis=1, training=is_training)
	x = tf.nn.relu(x, name='{}_relu'.format(name))
	return x


def DepthConv(x, kernel_shape, channel_mult=1, padding='SAME', stride=1, rate=1, data_format='NCHW',
				W_init=None, activation=tf.identity, name=None):
	# in_shape = x.get_shape().as_list()
	# if data_format == 'NHWC':
	# 	in_channel = in_shape[3]
	# 	stride_shape = [1, stride, stride, 1]
	# elif data_format == 'NCHW':
	# 	in_channel = in_shape[1]
	# 	stride_shape = [1, 1, stride, stride]
	# out_channel = in_channel * channel_mult

	# if W_init is None:
	# 	W_init = kernel_initializer
	kernel_shape = shape2d(kernel_shape)  # [kernel_shape, kernel_shape]
	# filter_shape = kernel_shape + [in_channel, channel_mult]
	#
	# W = tf.get_variable(name, filter_shape, initializer=W_init)
	# conv = tf.nn.depthwise_conv2d(x, W, stride_shape, padding=padding, rate=[rate, rate], data_format=data_format)
	conv = tf.keras.layers.DepthwiseConv2D(kernel_shape, padding=padding, strides=stride, data_format='channels_first',
										   use_bias=False, depth_multiplier=channel_mult)(x)

	return conv


def MyConv(name, x, filters, kernel_size, dw_size, ratio, padding='SAME', strides=1,
				data_format='NCHW', use_bias=False, is_training=False, activation=True):
	if data_format == 'NCHW':
		axis = 1
	else:
		axis = -1
	with tf.variable_scope(name):
		init_channels = math.ceil(filters / ratio)
		# if data_format == 'NCHW':
		x = tf.layers.Conv2D(init_channels, kernel_size, strides=strides, padding=padding,
							kernel_initializer=kernel_initializer, use_bias=use_bias, name=name + 'Conv', data_format='channels_first')(x)
		# else:
		# 	x = tf.layers.Conv2D(init_channels, kernel_size, strides=strides, padding=padding,
		# 					 	kernel_initializer=kernel_initializer, use_bias=use_bias, name=name + 'Conv', data_format='channels_last')(x)
		x = tf.layers.batch_normalization(x, training=is_training, name=name+'BN_1', axis=axis)
		x = tf.nn.relu(x, name=name+'Relu_1')

		if ratio == 1:
			return x
		dw1 = DepthConv(x, [dw_size, dw_size], channel_mult=ratio - 1, stride=1, data_format=data_format, name=name)
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


def ksize_for_squeezing(inputs, default_ksize=1024, data_format='NCHW'):
	"""Get the correct kernel size for squeezing an input tensor.
	"""
	shape = inputs.get_shape().as_list()
	kshape = shape[1:3] if data_format == 'NHWC' else shape[2:]
	if kshape[0] is None or kshape[1] is None:
		kernel_size_out = [default_ksize, default_ksize]
	else:
		kernel_size_out = [min(kshape[0], default_ksize),
						   min(kshape[1], default_ksize)]
	return kernel_size_out


def spatial_mean(inputs, scaling=None, keep_dims=False,
				 data_format='NCHW', scope=None):

	with tf.name_scope(scope, 'spatial_mean', [inputs]):
		axes = [1, 2] if data_format == 'NHWC' else [2, 3]
		net = tf.reduce_mean(inputs, axes, keep_dims=True)
		return net


def SELayer(x, out_dim, ratio, data_format='channels_first'):

	squeeze = tf.reduce_mean(x, axis=1, keepdims=True)
	# squeeze = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)

	excitation = tf.layers.Conv2D(int(out_dim / ratio), (1, 1), strides=(1, 1), kernel_initializer=kernel_initializer,
								  padding='same', data_format=data_format, use_bias=False)(squeeze)
	excitation = tf.nn.relu(excitation, name='relu')
	excitation = tf.layers.Conv2D(out_dim, 1, strides=1, kernel_initializer=kernel_initializer,
								  padding='same', data_format=data_format, use_bias=False)(excitation)
	# excitation = tf.clip_by_value(excitation, 0, 1, name='hsigmoid')
	excitation = tf.nn.sigmoid(excitation, name='sigmoid')
	# if data_format == 'channels_last':
	# 	excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
	# else:
	# excitation = tf.reshape(excitation, [-1, out_dim, 1, 1])

	scale = x * excitation

	return scale


def ghostnet_base(inputs,
				  final_endpoint=None,
				  min_depth=8,
				  depth_multiplier=1.0,
				  depth=1.0,
				  conv_defs=None,
				  output_stride=None,
				  dw_code=None,
				  ratio_code=None,
				  se=1,
				  scope=None,
				  is_training=False):

	""" By adjusting depth_multiplier can change the depth of network """

	def depth(d):
		d = max(int(d * depth_multiplier), min_depth)
		d = round(d / 4) * 4
		return d

	end_points = {}

	# Used to find thinned depths for each layer.
	if depth_multiplier <= 0:
		raise ValueError('depth_multiplier is not greater than zero.')

	if conv_defs is None:
		conv_defs = _CONV_DEFS_0

	if dw_code is None or len(dw_code) < len(conv_defs):
		dw_code = [3] * len(conv_defs)
	print('dw_code', dw_code)

	if ratio_code is None or len(ratio_code) < len(conv_defs):
		ratio_code = [2] * len(conv_defs)
	print('ratio_code', ratio_code)

	se_code = [x.se for x in conv_defs]
	print('se_code', se_code)

	if final_endpoint is None:
		final_endpoint = 'Conv2d_%d' % (len(conv_defs) - 1)

	if output_stride is not None and output_stride not in [8, 16, 32]:
		raise ValueError('Only allowed output_stride values are 8, 16, 32.')

	with tf.variable_scope(scope, 'MobilenetV2', [inputs]):

		# The current_stride variable keeps track of the output stride of the
		# activations, i.e., the running product of convolution strides up to the
		# current network layer. This allows us to invoke atrous convolution
		# whenever applying the next convolution would result in the activations
		# having output stride larger than the target output_stride.
		current_stride = 1

		# The atrous convolution rate parameter.
		rate = 1
		net = inputs
		in_depth = 3
		gi = 0
		for i, conv_def in enumerate(conv_defs):
			print('---')
			end_point_base = 'Conv2d_%d' % i
			if output_stride is not None and current_stride == output_stride:
				# If we have reached the target output_stride, then we need to employ
				# atrous convolution with stride=1 and multiply the atrous rate by the
				# current unit's stride for use in subsequent layers.
				layer_stride = 1
				layer_rate = rate
				rate *= conv_def.stride
			else:
				layer_stride = conv_def.stride
				layer_rate = 1
				current_stride *= conv_def.stride

			# change last bottleneck
			if i + 2 == len(conv_defs):			# remove last two head layers
				return [
					end_points['Conv2d_1_residual'],
					end_points['Conv2d_3_residual'],
					end_points['Conv2d_5_residual'],
					end_points['Conv2d_11_residual'],
					end_points['Conv2d_16_residual']
				]

			elif isinstance(conv_def, Conv):
				end_point = end_point_base
				net = ConvBlock(net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride,
								name='ConvBlock_{}'.format(i), is_training=is_training)
				end_points[end_point] = net

			# Bottleneck block.
			elif isinstance(conv_def, Bottleneck):
				# Stride > 1 or different depth: no residual part.
				if layer_stride == 1 and in_depth == conv_def.depth:
					res = net
				else:
					res = DepthConv(net, conv_def.kernel, stride=layer_stride, name='Bottleneck_block_{}_shortcut_dw'.format(i))
					res = tf.layers.batch_normalization(res, training=is_training, name='Bottleneck_block_{}_shortcut_dw_BN'.format(i), axis=1)
					res = ConvBlock(res, depth(conv_def.depth), (1, 1), (1, 1),
									name='Bottleneck_block_{}_shortcut_1x1'.format(i), is_training=is_training)

				# Increase depth with 1x1 conv.
				end_point = end_point_base + '_up_pointwise'
				net = MyConv('Bottleneck_block_{}_up_pointwise'.format(i), net, depth(in_depth * conv_def.factor), [1, 1],
							 dw_code[gi],
							 ratio_code[gi], strides=1, use_bias=False, is_training=is_training, activation=True)

				end_points[end_point] = net

				# Depthwise conv2d.
				if layer_stride > 1:
					end_point = end_point_base + '_depthwise'
					net = DepthConv(net, conv_def.kernel, stride=layer_stride, name='Bottleneck_block_{}_depthwise'.format(i))
					net = tf.layers.batch_normalization(net, training=is_training, name='Bottleneck_block_{}_depthwise_BN'.format(i), axis=1)
					end_points[end_point] = net
				# SE
				if se_code[i] > 0 and se > 0:
					end_point = end_point_base + '_se'
					net = SELayer(net, depth(in_depth * conv_def.factor), 4)
					end_points[end_point] = net

				# Downscale 1x1 conv.
				net = MyConv('Bottleneck_block_{}_down_pointwise'.format(i), net, depth(conv_def.depth), [1, 1], dw_code[gi],
							 ratio_code[gi], strides=1,  use_bias=False, is_training=is_training, activation=False)
				net = tf.layers.batch_normalization(net, training=is_training, name='Bottleneck_block_{}_down_pointwise_BN'.format(i), axis=1)

				gi += 1

				# Residual connection?
				end_point = end_point_base + '_residual'
				net = tf.add(res, net, name='Bottleneck_block_{}_Add'.format(i)) if res is not None else net
				end_points[end_point] = net

			in_depth = conv_def.depth
			# Final end point?

			if final_endpoint in end_points:

				return [
					end_points['Conv2d_1_residual'],
					end_points['Conv2d_3_residual'],
					end_points['Conv2d_5_residual'],
					end_points['Conv2d_11_residual'],
					end_points['Conv2d_16_residual']
				]


def DepthwiseConvBlock(input_tensor, kernel_size, strides, name, is_training=False):
	x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
								use_bias=False, name='{}_dconv'.format(name), data_format='channels_first')(input_tensor)
	x = tf.layers.batch_normalization(x, training=is_training, name='{}_bn'.format(name), axis=1)
	x = tf.nn.relu(x, name='{}_relu'.format(name))
	return x


def ConvBlock2(input_tensor, num_channels, kernel_size, strides, name, is_training=False):
	x = tf.layers.conv2d(input_tensor,num_channels, kernel_size=kernel_size, strides=strides, padding='same',
					   use_bias=False, name='{}_conv'.format(name), data_format='channels_first')
	x = tf.layers.batch_normalization(x, training=is_training, name='{}_bn'.format(name), axis=1)
	x = tf.nn.relu(x, name='{}_relu'.format(name))
	return x


def bifpn_segmentation_head(features, num_channels, scope, is_training=False, reuse=None):
	""" BiFPN """

	with tf.variable_scope(scope, reuse=reuse):
		P3_in, P4_in, P5_in, P6_in, P7_in = features
		P3_in = ConvBlock2(P3_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P3')
		P4_in = ConvBlock2(P4_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P4')
		P5_in = ConvBlock2(P5_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P5')
		P6_in = ConvBlock2(P6_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P6')
		P7_in = ConvBlock2(P7_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P7')
		# upsample
		P7_U = keras.layers.UpSampling2D(interpolation='nearest', data_format='channels_first')(P7_in)
		# P6_td = keras.layers.Add()([P7_U, P6_in])
		P6_td = P7_U + P6_in
		P6_td = DepthwiseConvBlock(P6_td, kernel_size=3, strides=1, is_training=is_training, name='BiFPN_U_P6')
		P6_U = keras.layers.UpSampling2D(interpolation='nearest', data_format='channels_first')(P6_td)
		# P5_td = keras.layers.Add()([P6_U, P5_in])
		P5_td = P6_U + P5_in
		P5_td = DepthwiseConvBlock(P5_td, kernel_size=3, strides=1, is_training=is_training, name='BiFPN_U_P5')
		P5_U = keras.layers.UpSampling2D(interpolation='nearest', data_format='channels_first')(P5_td)
		# P4_td = keras.layers.Add()([P5_U, P4_in])
		P4_td = P5_U + P4_in
		P4_td = DepthwiseConvBlock(P4_td, kernel_size=3, strides=1, is_training=is_training, name='BiFPN_U_P4')
		P4_U = keras.layers.UpSampling2D(interpolation='nearest', data_format='channels_first')(P4_td)
		# P3_out = keras.layers.Add()([P4_U, P3_in])
		P3_out = P4_U + P3_in
		P3_out = DepthwiseConvBlock(P3_out, kernel_size=3, strides=1, is_training=is_training, name='BiFPN_U_P3')
		# downsample
		# P3_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format='channels_first')(P3_out)
		P3_D = tf.layers.max_pooling2d(P3_out, pool_size=(2, 2), strides=(2, 2), data_format='channels_first',
									   name='max_pool0')
		# P4_out = keras.layers.Add()([P3_D, P4_td, P4_in])
		P4_out = P3_D + P4_td + P4_in
		P4_out = DepthwiseConvBlock(P4_out, kernel_size=3, strides=1, is_training=is_training, name='BiFPN_D_P4')
		# P4_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format='channels_first')(P4_out)
		P4_D = tf.layers.max_pooling2d(P4_out, pool_size=(2, 2), strides=(2, 2), data_format='channels_first', name='max_pool1')
		# P5_out = keras.layers.Add()([P4_D, P5_td, P5_in])
		P5_out = P4_D + P5_td + P5_in
		P5_out = DepthwiseConvBlock(P5_out, kernel_size=3, strides=1, is_training=is_training, name='BiFPN_D_P5')
		# P5_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format='channels_first')(P5_out)
		P5_D = tf.layers.max_pooling2d(P5_out, pool_size=(2, 2), strides=(2, 2), data_format='channels_first',
									   name='max_pool2')
		# P6_out = keras.layers.Add()([P5_D, P6_td, P6_in])
		P6_out = P5_D + P6_td + P6_in
		P6_out = DepthwiseConvBlock(P6_out, kernel_size=3, strides=1, is_training=is_training, name='BiFPN_D_P6')
		# P6_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format='channels_first')(P6_out)
		P6_D = tf.layers.max_pooling2d(P6_out, pool_size=(2, 2), strides=(2, 2), data_format='channels_first',
									   name='max_pool3')
		# P7_out = keras.layers.Add()([P6_D, P7_in])
		P7_out = P6_D + P7_in
		P7_out = DepthwiseConvBlock(P7_out, kernel_size=3, strides=1, is_training=is_training, name='BiFPN_D_P7')

	return P3_out, P4_out, P5_out, P6_out, P7_out


def decision_head(x, y, class_num, scope, keep_dropout_head, training=False, reuse=None, drop_rate=0.2):

	with tf.variable_scope(scope, reuse=reuse):

		x = tf.concat([x, y], axis=1)
		x = tf.layers.conv2d(x, 16, (3, 3), strides=(2, 2), padding='same', name='dec_conv2d1', data_format='channels_first', use_bias=False)
		x = tf.layers.batch_normalization(x, training=training, name='dec_bn1', axis=1)
		x = tf.nn.relu(x, name='dec_relu1')
		x = tf.layers.conv2d(x, 16, (3, 3), strides=(1, 1), padding='same', name='dec_conv2d2', data_format='channels_first', use_bias=False)
		x = tf.layers.batch_normalization(x, training=training, name='dec_bn2', axis=1)
		x = tf.nn.relu(x, name='dec_relu2')

		x = tf.layers.conv2d(x, 32, (3, 3), strides=(2, 2), padding='same', name='dec_conv2d3', data_format='channels_first', use_bias=False)
		x = tf.layers.batch_normalization(x, training=training, name='dec_bn3', axis=1)
		x = tf.nn.relu(x, name='dec_relu3')
		x = tf.layers.conv2d(x, 32, (3, 3), strides=(1, 1), padding='same', name='dec_conv2d4', data_format='channels_first', use_bias=False)
		x = tf.layers.batch_normalization(x, training=training, name='dec_bn4', axis=1)
		x = tf.nn.relu(x, name='dec_relu4')

		x = tf.layers.conv2d(x, 32, (3, 3), strides=(2, 2), padding='same', name='dec_conv2d5', data_format='channels_first', use_bias=False)

		# de_max_po = tf.keras.layers.GlobalMaxPool2D(data_format='channels_first', name='GlobalMaxPooling1')(x)
		#
		# de_avg_po = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first', name='GlobalAveragePooling1')(x)
		#
		# seg_max_po = tf.keras.layers.GlobalMaxPool2D(data_format='channels_first', name='GlobalMaxPooling2')(y)
		#
		# seg_avg_po = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first', name='GlobalAveragePooling2')(y)

		de_max_po = math_ops.reduce_mean(x, [2, 3], name='pool4', keepdims=True)
		de_avg_po = math_ops.reduce_max(x, [2, 3], name='pool5', keepdims=True)
		seg_max_po = math_ops.reduce_mean(y, [2, 3], name='pool6', keepdims=True)
		seg_avg_po = math_ops.reduce_max(y, [2, 3], name='pool7', keepdims=True)

		# de_max_po = tf.layers.Flatten(name='dec_flatten1', data_format='channels_first')(de_max_po)
		# de_avg_po = tf.layers.Flatten(name='dec_flatten2', data_format='channels_first')(de_avg_po)
		# seg_max_po = tf.layers.Flatten(name='dec_flatten3', data_format='channels_first')(seg_max_po)
		# seg_avg_po = tf.layers.Flatten(name='dec_faltten4', data_format='channels_first')(seg_avg_po)

		x = tf.concat([de_max_po, de_avg_po, seg_max_po, seg_avg_po], axis=1)
		x = tf.squeeze(x, axis=[2, 3])
		# if keep_dropout_head:
		# 	x = tf.nn.dropout(x, keep_prob=1-drop_rate)
		x = tf.layers.dense(x, class_num, name='dense', use_bias=False)

		return x


"""---------------------------------------Data 	Manager----------------------------------------------------"""

class DataManager(object):
	def __init__(self, imageList, maskList, shuffle=True):

		self.shuffle = shuffle
		self.image_list = imageList
		self.mask_list = maskList
		self.data_size = len(imageList)
		self.batch_size = 1
		self.number_batch = int(np.floor(len(self.image_list) / self.batch_size))
		self.next_batch = self.get_next()

	def get_next(self):
		dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.float32, tf.string))
		dataset = dataset.repeat()

		dataset = dataset.batch(self.batch_size)
		iterator = dataset.make_one_shot_iterator()
		out_batch = iterator.get_next()
		return out_batch

	def generator(self):
		rand_index = np.arange(len(self.image_list))
		np.random.shuffle(rand_index)
		for index in range(len(self.image_list)):
			image_path = self.image_list[rand_index[index]]
			mask_path = self.mask_list[rand_index[index]]
			if image_path.split('/')[-1].split('_')[0] == 'n':
				label = np.array([0.0])
			else:
				label = np.array([1.0])

			image, mask = self.read_data(image_path, mask_path)
			image = image / 255
			mask = mask / 255
			mask = (np.array(mask[:, :, np.newaxis]))
			image = (np.array(image[np.newaxis, :, :]))
			# image = np.transpose(image, [2, 0, 1])

			yield image, mask, label, image_path

	def read_data(self, image_path, mask_path):

		img = cv2.imread(image_path, 0)  # /255.#read the gray image
		img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
		msk = cv2.imread(mask_path, 0)  # /255.#read the gray image

		msk = cv2.resize(msk, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
		return img, msk


def listData_train(data_dir):
	image_dirs = [x[2] for x in os.walk(data_dir + 'train_image/')][0]
	images_train = []
	masks_train = []

	for i in range(len(image_dirs)):
		image_dir = image_dirs[i]
		image_path = data_dir + 'train_image/' + image_dir
		mask_path = data_dir + 'train_mask/' + image_dir
		images_train.append(image_path)
		masks_train.append(mask_path)
	return images_train, masks_train


def listData_val(data_dir):
	image_dirs = [x[2] for x in os.walk(data_dir + 'val_image/')][0]
	images_val = []
	masks_val = []

	for i in range(len(image_dirs)):
		image_dir = image_dirs[i]
		image_path = data_dir + 'val_image/' + image_dir
		mask_path = data_dir + 'val_mask/' + image_dir
		images_val.append(image_path)
		masks_val.append(mask_path)
	return images_val, masks_val

def listData_test(data_dir):
	image_dirs = [x[2] for x in os.walk(data_dir + 'val_image/')][0]
	images_val = []
	masks_val = []

	for i in range(len(image_dirs)):
		image_dir = image_dirs[i]
		image_path = data_dir + 'test_image/' + image_dir
		mask_path = data_dir + 'test_mask/' + image_dir
		images_val.append(image_path)
		masks_val.append(mask_path)
	return images_val, masks_val


data_dir = "F:/CODES/FAST-SCNN/DATA/1pzt/"
image_list_train, mask_list_train = listData_train(data_dir)
image_list_valid, mask_list_valid = listData_val(data_dir)
image_list_test, mask_list_test = listData_test(data_dir)

DataManager_train = DataManager(image_list_train, mask_list_train)
DataManager_valid = DataManager(image_list_valid, mask_list_valid, shuffle=False)
DataManager_test = DataManager(image_list_test, mask_list_test, shuffle=False)


"""------------------------------Model 		Loss 	Optimizer------------------------------------------------"""

checkPoint_dir = "checkpoint"
sess = tf.Session()
with sess.as_default():
	image_input = tf.placeholder(tf.float32,
								 shape=(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1]),
								 name='Image')

	label = tf.placeholder(tf.float32, shape=(1, 1), name='Label')
	mask = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), name='mask')
	is_training_seg = tf.placeholder(tf.bool, name='is_training_seg')
	is_training_dec = tf.placeholder(tf.bool, name='is_training_dec')

	feature_list = ghostnet_base(image_input, scope='segmentation', dw_code=None, ratio_code=None,
								 se=1, min_depth=8, depth=1, depth_multiplier=0.5, conv_defs=None,
								 is_training=is_training_seg)

	P3_out, P4_out, P5_out, P6_out, P7_out = bifpn_segmentation_head(feature_list, 64, scope='segmentation',
																	 is_training=is_training_seg)
	with tf.variable_scope('segmentation'):
		P3 = tf.layers.conv2d(P3_out, 1, (1, 1), (1, 1), use_bias=False, name='P3', data_format='channels_first')
		seg_out = tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')(P3)
		# seg_out = tf.image.resize_images(P3, (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True)
		P4 = tf.layers.conv2d(P4_out, 1, (1, 1), (1, 1), use_bias=False, name='P4', data_format='channels_first')
		P5 = tf.layers.conv2d(P5_out, 1, (1, 1), (1, 1), use_bias=False, name='P5', data_format='channels_first')
		P6 = tf.layers.conv2d(P6_out, 1, (1, 1), (1, 1), use_bias=False, name='P6', data_format='channels_first')
		P7 = tf.layers.conv2d(P7_out, 1, (1, 1), (1, 1), use_bias=False, name='P7', data_format='channels_first')
		feature = [P3_out, P3]

	dec_out = decision_head(feature[0], feature[1], class_num=1, scope='decision',
							keep_dropout_head=False,
							training=is_training_dec)
	decision_out = tf.nn.sigmoid(dec_out, name='decision_out')


	segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=seg_out, labels=tf.transpose(mask, [0, 3, 1, 2]))) + \
	tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P4, labels=tf.transpose(tf.image.resize_images(mask, (IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4)), [0, 3, 1, 2]))) + \
	tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P5, labels=tf.transpose(tf.image.resize_images(mask, (IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8)), [0, 3, 1, 2]))) + \
	tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P6, labels=tf.transpose(tf.image.resize_images(mask, (IMAGE_SIZE[0] // 16, IMAGE_SIZE[1] // 16)), [0, 3, 1, 2]))) + \
	tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P7, labels=tf.transpose(tf.image.resize_images(mask, (IMAGE_SIZE[0] // 32, IMAGE_SIZE[1] // 32)), [0, 3, 1, 2])))

	decision_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_out, labels=label)

	train_segment_var_list = [v for v in tf.trainable_variables() if 'segmentation' in v.name]
	train_decision_var_list = [v for v in tf.trainable_variables() if 'decision' in v.name]

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	update_ops_segment = [v for v in update_ops if 'segmentation' in v.name]

	update_ops_decision = [v for v in update_ops if 'decision' in v.name]

	optimizer_segmentation = tf.train.AdamOptimizer(0.001)
	optimizer_decision = tf.train.AdamOptimizer(0.001)

	with tf.control_dependencies(update_ops_segment):
		optimize_segment = optimizer_segmentation.minimize(segmentation_loss, var_list=train_segment_var_list)

	with tf.control_dependencies(update_ops_decision):
		optimize_decision = optimizer_decision.minimize(decision_loss, var_list=train_decision_var_list)

	sess.run(tf.global_variables_initializer())
	var_list = tf.trainable_variables()
	g_list = tf.global_variables()

	bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
	bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
	var_list += bn_moving_vars
	saver = tf.train.Saver(var_list)
	ckpt = tf.train.latest_checkpoint(checkPoint_dir)
	if ckpt:
		step = int(ckpt.split('-')[1])
		saver.restore(sess, ckpt)
		print('Restoring from epoch:{}'.format(step))

"""------------------------------------Train 	  Segmentation------------------------------------------------"""


with sess.as_default():

	sess.run(tf.global_variables_initializer())
	seg_epochs = 10
	print('Start training segmentation for {} epoches'.format(seg_epochs))
	best_loss = 10000
	for i in range(seg_epochs):
		print('Epoch {}:'.format(i))
		with tqdm(total=DataManager_train.number_batch) as pbar:
			iter_loss = 0.0
			num_step = 0.0
			accuracy = 0.0
			for batch in range(DataManager_train.number_batch):
				img_batch, mask_batch, label_batch, _ = sess.run(DataManager_train.next_batch)

				_, loss_value_batch = sess.run([optimize_segment,
												 segmentation_loss],
												feed_dict={image_input: img_batch,
														   mask: mask_batch,
														   label: label_batch,
														   is_training_seg: True,
														   is_training_dec: False})
				iter_loss += loss_value_batch
				num_step = num_step + 1
				pbar.update(1)
		pbar.close()
		iter_loss /= num_step

		print('start validing segmentation')
		val_loss = 0.0
		num_step = 0.0

		for batch in range(DataManager_valid.number_batch):
			img_batch, mask_batch, label_batch, _ = sess.run(DataManager_valid.next_batch)

			total_loss_value_batch = sess.run(segmentation_loss,
											  feed_dict={image_input: img_batch,
														 mask: mask_batch,
														 label: label_batch,
														 is_training_seg: False,
														 is_training_dec: False})
			num_step = num_step + 1
			val_loss += total_loss_value_batch
		val_loss /= num_step

		print('train_loss:{},   val_loss:{}'.format(iter_loss, val_loss))
		saver.save(sess, os.path.join(checkPoint_dir, 'ckp'), global_step=seg_epochs)

"""----------------------------------Train		Decision---------------------------------------------------"""

with sess.as_default():
	dec_epochs = 5
	with sess.as_default():
		print('Start training decision for {} epoches'.format(dec_epochs))
		best_loss = 10000
		for i in range(seg_epochs, dec_epochs+seg_epochs):
			print('Epoch {}:'.format(i))
			with tqdm(total=DataManager_train.number_batch) as pbar:
				iter_loss = 0.0
				num_step = 0.0
				acc = 0
				for batch in range(DataManager_train.number_batch):
					img_batch, mask_batch, label_batch, _ = sess.run(DataManager_train.next_batch)

					_, loss_value_batch, decision = sess.run([optimize_decision,
															  decision_loss,
															  decision_out],
												   feed_dict={image_input: img_batch,
															  mask: mask_batch,
															  label: label_batch,
															  is_training_seg: False,
															  is_training_dec: True})
					if decision[0][0] >= 0.5 and label_batch[0][0] == 1:
						step_accuracy = 1
					elif decision[0][0] < 0.5 and label_batch[0][0] == 0:
						step_accuracy = 1
					else:
						step_accuracy = 0
					acc = acc + step_accuracy
					iter_loss += loss_value_batch
					num_step = num_step + 1
					pbar.update(1)
				iter_loss /= num_step
				acc /= num_step
			pbar.close()

			print('start validing decision')
			val_loss = 0.0
			num_step = 0.0
			accuracy = 0.0
			for batch in range(DataManager_valid.number_batch):
				img_batch, mask_batch, label_batch, _ = sess.run(DataManager_valid.next_batch)

				total_loss_value_batch, decision = sess.run([decision_loss, decision_out],
												  feed_dict={image_input: img_batch,
															 mask: mask_batch,
															 label: label_batch,
															 is_training_seg: False,
															 is_training_dec: False})
				if decision[0][0] >= 0.5 and label_batch[0][0] == 1:
					step_accuracy = 1
				elif decision[0][0] < 0.5 and label_batch[0][0] == 0:
					step_accuracy = 1
				else:
					step_accuracy = 0
				accuracy = accuracy + step_accuracy
				num_step = num_step + 1
				val_loss += total_loss_value_batch
			val_loss /= num_step
			accuracy /= num_step

			print('train_loss:{},   val_loss:{}, 	train_acc:{}, 	val_acc:{}'.format(iter_loss, val_loss, acc, accuracy))
			saver.save(sess, os.path.join(checkPoint_dir, 'ckp'), global_step=dec_epochs+seg_epochs)


#
# """-------------------------------Restore from checkpoint Rmove Loss and save to Pb------------------------------"""
#
# checkPoint_dir = 'checkpoint'
# session = tf.Session()
# with session.as_default():
# 	image_input = tf.placeholder(tf.float32, shape=(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1]), name='Image')
# 	is_training_seg = False
# 	is_training_dec = False
#
# 	feature_list = ghostnet_base(image_input, scope='segmentation', dw_code=None, ratio_code=None,
# 								 se=1, min_depth=8, depth=1, depth_multiplier=0.5, conv_defs=None,
# 								 is_training=is_training_seg)
#
# 	P3_out, P4_out, P5_out, P6_out, P7_out = bifpn_segmentation_head(feature_list, 64, scope='segmentation',
# 																	 is_training=is_training_seg)
# 	with tf.variable_scope('segmentation'):
# 		P3 = tf.layers.conv2d(P3_out, 1, (1, 1), (1, 1), use_bias=False, name='P3', data_format='channels_first')
# 		seg_out = tf.keras.layers.UpSampling2D((2, 2), data_format='channels_first')(P3)
# 		# seg_out = tf.image.resize_images(P3, (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True)
# 		P4 = tf.layers.conv2d(P4_out, 1, (1, 1), (1, 1), use_bias=False, name='P4', data_format='channels_first')
# 		P5 = tf.layers.conv2d(P5_out, 1, (1, 1), (1, 1), use_bias=False, name='P5', data_format='channels_first')
# 		P6 = tf.layers.conv2d(P6_out, 1, (1, 1), (1, 1), use_bias=False, name='P6', data_format='channels_first')
# 		P7 = tf.layers.conv2d(P7_out, 1, (1, 1), (1, 1), use_bias=False, name='P7', data_format='channels_first')
# 		feature = [P3_out, P3]
#
# 	dec_out = decision_head(feature[0], feature[1], class_num=1, scope='decision',
# 							keep_dropout_head=False,
# 							training=is_training_dec)
# 	decision_out = tf.nn.sigmoid(dec_out, name='decision_out')
#
# 	var_list = tf.trainable_variables()
# 	g_list = tf.global_variables()
#
# 	bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
# 	bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
# 	var_list += bn_moving_vars
# 	saver = tf.train.Saver(var_list)
# 	ckpt = tf.train.latest_checkpoint(checkPoint_dir)
# 	if ckpt:
# 		step = int(ckpt.split('-')[1])
# 		saver.restore(session, ckpt)
# 		print('Restoring from epoch:{}'.format(step))
#
#
# def save_PbModel(session, __pb_model_path):
# 	input_name = "Image"
# 	output_name = "decision_out"
# 	output_node_names = [input_name, output_name]
# 	output_graph_def = tf.graph_util.convert_variables_to_constants(session,
# 																	session.graph_def,
# 																	output_node_names)
# 	# output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)
#
# 	if not os.path.exists(__pb_model_path):
# 		os.makedirs(__pb_model_path)
# 	pbpath = os.path.join(__pb_model_path, 'tensorrt.pb')
# 	with tf.gfile.GFile(pbpath, mode='wb') as f:
# 		f.write(output_graph_def.SerializeToString())
#
#
# pb_Model_dir = "pbMode"
# save_PbModel(session, pb_Model_dir)

# '''--------------------------------------------Test    Pb-------------------------------------------------------'''
#
# class DataManager(object):
# 	def __init__(self, imageList, maskList, shuffle=True):
#
# 		self.shuffle = shuffle
# 		self.image_list = imageList
# 		self.mask_list = maskList
# 		self.data_size = len(imageList)
# 		self.batch_size = 1
# 		self.number_batch = int(np.floor(len(self.image_list) / self.batch_size))
# 		self.next_batch = self.get_next()
#
# 	def get_next(self):
# 		dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.string))
# 		dataset = dataset.repeat()
#
# 		dataset = dataset.batch(self.batch_size)
# 		iterator = dataset.make_one_shot_iterator()
# 		out_batch = iterator.get_next()
# 		return out_batch
#
# 	def generator(self):
# 		rand_index = np.arange(len(self.image_list))
# 		np.random.shuffle(rand_index)
# 		for index in range(len(self.image_list)):
# 			image_path = self.image_list[rand_index[index]]
# 			mask_path = self.mask_list[rand_index[index]]
# 			if image_path.split('/')[-1].split('_')[0] == 'n':
# 				label = np.array([0.0])
# 			else:
# 				label = np.array([1.0])
#
# 			image, mask = self.read_data(image_path, mask_path)
# 			image = image / 255
# 			mask = mask / 255
# 			mask = (np.array(mask[:, :, np.newaxis]))
# 			image = np.transpose(image, [2, 0, 1])
#
# 			yield image, label, image_path
#
# 	def read_data(self, image_path, mask_path):
#
# 		img = cv2.imread(image_path, 1)  # /255.#read the gray image
# 		img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
# 		msk = cv2.imread(mask_path, 0)  # /255.#read the gray image
#
# 		msk = cv2.resize(msk, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
# 		return img, msk
#
#
# def listData_train(data_dir):
# 	image_dirs = [x[2] for x in os.walk(data_dir + 'train_image/')][0]
# 	images_train = []
# 	masks_train = []
#
# 	for i in range(len(image_dirs)):
# 		image_dir = image_dirs[i]
# 		image_path = data_dir + 'train_image/' + image_dir
# 		mask_path = data_dir + 'train_mask/' + image_dir
# 		images_train.append(image_path)
# 		masks_train.append(mask_path)
# 	return images_train, masks_train
#
#
# def listData_val(data_dir):
# 	image_dirs = [x[2] for x in os.walk(data_dir + 'val_image/')][0]
# 	images_val = []
# 	masks_val = []
#
# 	for i in range(len(image_dirs)):
# 		image_dir = image_dirs[i]
# 		image_path = data_dir + 'val_image/' + image_dir
# 		mask_path = data_dir + 'val_mask/' + image_dir
# 		images_val.append(image_path)
# 		masks_val.append(mask_path)
# 	return images_val, masks_val
#
# def listData_test(data_dir):
# 	image_dirs = [x[2] for x in os.walk(data_dir + 'test_image/')][0]
# 	images_val = []
# 	masks_val = []
#
# 	for i in range(len(image_dirs)):
# 		image_dir = image_dirs[i]
# 		image_path = data_dir + 'test_image/' + image_dir
# 		mask_path = data_dir + 'test_mask/' + image_dir
# 		images_val.append(image_path)
# 		masks_val.append(mask_path)
# 	return images_val, masks_val
#
#
# data_dir = "F:/CODES/FAST-SCNN/DATA/1pzt/"
# image_list_train, mask_list_train = listData_train(data_dir)
# image_list_valid, mask_list_valid = listData_val(data_dir)
# image_list_test, mask_list_test = listData_test(data_dir)
#
# DataManager_train = DataManager(image_list_train, mask_list_train)
# DataManager_valid = DataManager(image_list_valid, mask_list_valid, shuffle=False)
# DataManager_test = DataManager(image_list_test, mask_list_test, shuffle=False)
#
# def test(sess, dataset):
# 	with sess.as_default():
#
# 		input_image = sess.graph.get_tensor_by_name('Image:0')
# 		decision_out = sess.graph.get_tensor_by_name('decision_out:0')
# 		DataManager = dataset
# 		num_step = 0.0
# 		accuracy = 0.0
# 		for batch in range(DataManager.number_batch):
# 			img_batch, label_batch, _ = sess.run(DataManager.next_batch)
# 			start = timer()
# 			decision = sess.run(decision_out, feed_dict={input_image: img_batch})
# 			end = timer()
# 			print(end-start)
# 			if decision[0][0] >= 0.5 and label_batch[0][0] == 1:
# 				step_accuracy = 1
# 			elif decision[0][0] < 0.5 and label_batch[0][0] == 0:
# 				step_accuracy = 1
# 			else:
# 				step_accuracy = 0
# 			accuracy = accuracy + step_accuracy
# 			num_step = num_step + 1
# 		accuracy /= num_step
#
# 		return accuracy
#
#
# from tensorflow.python.platform import gfile
# pb_file_path = './pbMode/frozen_inference_graph.pb'
#
# sess = tf.Session()
# with gfile.FastGFile(pb_file_path, 'rb') as f:
# 	graph_def = tf.GraphDef()
# 	graph_def.ParseFromString(f.read())
# 	sess.graph.as_default()
# 	tf.import_graph_def(graph_def, name='')
# sess.run(tf.global_variables_initializer())
#
# train_acc = test(sess, DataManager_train)
# val_acc = test(sess, DataManager_valid)
# test_acc = test(sess, DataManager_test)
#
# print('train_accuracy = {},   val_accuracy = {},   test_accuracy = {}'.format(train_acc, val_acc, test_acc))
#
