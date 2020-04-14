from tensorflow import keras
from collections import namedtuple
import tensorflow as tf
from GhostNetModule import ConvBlock
from GhostNetModule import GhostModule as MyConv

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


def ksize_for_squeezing(inputs, default_ksize=1024, data_format='NHWC'):
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


def DepthConv(x, kernel_shape, padding='SAME', stride=1, data_format='NHWC', W_init=None, name=None):

	with tf.variable_scope(name):
		in_shape = x.get_shape().as_list()
		in_channel = in_shape[3]
		stride_shape = [1, stride, stride, 1]
		out_channel = in_channel
		channel_mult = out_channel // in_channel

		if W_init is None:
			W_init = kernel_initializer
		kernel_shape = shape2d(kernel_shape)  # [kernel_shape, kernel_shape]
		filter_shape = kernel_shape + [in_channel, channel_mult]

		W = tf.get_variable('W', filter_shape, initializer=W_init)
		conv = tf.nn.depthwise_conv2d(x, W, stride_shape, padding=padding, data_format=data_format)
		return conv


def spatial_mean(inputs, scaling=None, keep_dims=False,
				 data_format='NHWC', scope=None):
	"""Average tensor along spatial dimensions.

	Args:
	  inputs: Input tensor;
	  keep_dims: Keep spatial dimensions?
	  data_format: NHWC or NCHW.
	"""
	with tf.name_scope(scope, 'spatial_mean', [inputs]):
		axes = [1, 2] if data_format == 'NHWC' else [2, 3]
		net = tf.reduce_mean(inputs, axes, keep_dims=True)
		return net


def SELayer(x, out_dim, ratio):

	squeeze = spatial_mean(x, keep_dims=True, scope='global_pool')

	excitation = tf.layers.Conv2D(int(out_dim / ratio), (1, 1), strides=(1, 1), kernel_initializer=kernel_initializer,
								  padding='same')(squeeze)
	excitation = tf.nn.relu(excitation, name='relu')
	excitation = tf.layers.Conv2D(out_dim, 1, strides=1, kernel_initializer=kernel_initializer,
								  padding='same')(excitation)
	excitation = tf.clip_by_value(excitation, 0, 1, name='hsigmoid')
	excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
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
			# 	end_point = end_point_base
			#
			# 	net = ConvBlock(net, depth(conv_def.depth), (1, 1), (1, 1),
			# 					name='ConvBlock_{}'.format(i), is_training=is_training)
			#
			# 	ksize = ksize_for_squeezing(net, 1024)
			# 	net = tf.layers.AveragePooling2D(pool_size=ksize, strides=(1, 1), padding='VALID')(net)
			#
			# 	end_points[end_point] = net

			# Normal conv2d.
			# elif i + 1 == len(conv_defs):
			# 	end_point = end_point_base
			# 	net = ConvBlock(net, 1280, conv_def.kernel, stride=conv_def.stride,
			# 					name='ConvBlock_{}'.format(i), is_training=is_training)
			# 	end_points[end_point] = net

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
					res = DepthConv(net, conv_def.kernel, stride=layer_stride, data_format='NHWC', name='Bottleneck_block_{}_shortcut_dw'.format(i))
					res = tf.layers.batch_normalization(res, training=is_training, name='Bottleneck_block_{}_shortcut_dw_BN'.format(i))
					res = ConvBlock(res, depth(conv_def.depth), (1, 1), (1, 1),
									name='Bottleneck_block_{}_shortcut_1x1'.format(i), is_training=is_training)

				# Increase depth with 1x1 conv.
				end_point = end_point_base + '_up_pointwise'
				net = MyConv('Bottleneck_block_{}_up_pointwise'.format(i), net, depth(in_depth * conv_def.factor), [1, 1],
							 dw_code[gi],
							 ratio_code[gi], strides=1, data_format='NHWC', use_bias=False, is_training=is_training, activation=True)

				end_points[end_point] = net

				# Depthwise conv2d.
				if layer_stride > 1:
					end_point = end_point_base + '_depthwise'
					net = DepthConv(net, conv_def.kernel, stride=layer_stride, data_format='NHWC', name='Bottleneck_block_{}_depthwise'.format(i))
					net = tf.layers.batch_normalization(net, training=is_training, name='Bottleneck_block_{}_depthwise_BN'.format(i))
					end_points[end_point] = net
				# SE
				if se_code[i] > 0 and se > 0:
					end_point = end_point_base + '_se'
					net = SELayer(net, depth(in_depth * conv_def.factor), 4)
					end_points[end_point] = net

				# Downscale 1x1 conv.
				net = MyConv('Bottleneck_block_{}_down_pointwise'.format(i), net, depth(conv_def.depth), [1, 1], dw_code[gi],
							 ratio_code[gi], strides=1, data_format='NHWC', use_bias=False, is_training=is_training, activation=False)
				net = tf.layers.batch_normalization(net, training=is_training, name='Bottleneck_block_{}_down_pointwise_BN'.format(i))

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


if __name__ == '__main__':
	inputs = keras.layers.Input((928, 320, 3))

	end_points = ghostnet_base(inputs, scope='segmentation', dw_code=None, ratio_code=None,
									se=1, min_depth=8, depth=1.0, depth_multiplier=1, conv_defs=None)
	for i in end_points:
		print(i)

	pass



