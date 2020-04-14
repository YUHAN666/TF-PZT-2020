import string
import collections
import tensorflow as tf
from tensorflow.python.ops import math_ops
from six.moves import xrange
import math
from tensorflow.python.framework import ops
from tensorflow import keras as keras
from config import IMAGE_SIZE


BlockArgs = collections.namedtuple('BlockArgs', [
	'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
	'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

IMAGE_HEIGHT = IMAGE_SIZE[0]
IMAGE_WIDTH = IMAGE_SIZE[1]

# DEFAULT_BLOCKS_ARGS = [
#     # BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
#     #           expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
#     BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
#               expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
#     BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
#               expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
#     BlockArgs(kernel_size=3, num_repeat=2, input_filters=40, output_filters=80,
#               expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
#     # BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
#     #           expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
#     BlockArgs(kernel_size=5, num_repeat=2, input_filters=112, output_filters=192,
#               expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
#     # BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
#     #           expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
# ]

DEFAULT_BLOCKS_ARGS = [
	BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
			  expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
	BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
			  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
	BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
			  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
	BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
			  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
	BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
			  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
	BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
			  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
	BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
			  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]


CONV_KERNEL_INITIALIZER = {
	'class_name': 'VarianceScaling',
	'config': {
		'scale': 2.0,
		'mode': 'fan_out',
		# EfficientNet actually uses an untruncated normal distribution for
		# initializing conv layers, but keras.initializers.VarianceScaling use
		# a truncated distribution.
		# We decided against a custom initializer for better serializability.
		'distribution': 'normal'
	}
}

DENSE_KERNEL_INITIALIZER = {
	'class_name': 'VarianceScaling',
	'config': {
		'scale': 1. / 3.,
		'mode': 'fan_out',
		'distribution': 'uniform'
	}
}


def swish(x, name=None):
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


def round_filters(filters, width_coefficient, depth_divisor):
	"""Round number of filters based on width multiplier."""

	filters *= width_coefficient
	new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
	new_filters = max(depth_divisor, new_filters)
	# Make sure that round down does not go down by more than 10%.
	if new_filters < 0.9 * filters:
		new_filters += depth_divisor
	return int(new_filters)


def round_repeats(repeats, depth_coefficient):
	"""Round number of repeats based on depth multiplier."""

	return int(math.ceil(depth_coefficient * repeats))


def Squeeze_excitation_layer(input_x, out_dim, se_dim, name):
	with tf.name_scope(name):

		squeeze = math_ops.reduce_mean(input_x, [1, 2], name=name+'_gap', keepdims=True)

		excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=se_dim, kernel_initializer=DENSE_KERNEL_INITIALIZER, name=name+'_fully_connected1')

		excitation = tf.nn.relu(excitation, name=name+'_swish')

		excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=out_dim, kernel_initializer=DENSE_KERNEL_INITIALIZER, name=name+'_fully_connected2')

		excitation = tf.nn.sigmoid(excitation, name=name+'_sigmoid')

		excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

		scale = input_x * excitation

		return scale

# def Squeeze_excitation_layer(input_x, out_dim, se_dim, name):
#     with tf.name_scope(name):
#
#         squeeze = tf.layers.average_pooling2d(input_x, pool_size=(input_x.shape[1], input_x.shape[2]),
#                                               strides=(1, 1), name=name+'_gap')
#
#         excitation = tf.layers.conv2d(squeeze, se_dim, (1, 1), (1, 1), use_bias=True,
#                                       kernel_initializer=CONV_KERNEL_INITIALIZER, name=name+'_fully_connected1')
#
#         excitation = tf.nn.relu(excitation, name=name+'_swish')
#
#         excitation = tf.layers.conv2d(excitation, out_dim, (1, 1), (1, 1), use_bias=True,
#                                       kernel_initializer=CONV_KERNEL_INITIALIZER, name=name+'_fully_connected2')
#
#         excitation = tf.nn.sigmoid(excitation, name=name+'_sigmoid')
#
#         excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
#
#         scale = input_x * excitation
#
#         return scale


def mb_conv_block(inputs, block_args, training, keep_dropout_backbone, drop_rate=None, prefix='', ):
	"""Mobile Inverted Residual Bottleneck."""

	has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
	# Expansion phase
	filters = block_args.input_filters * block_args.expand_ratio

	if block_args.expand_ratio != 1:
		x = tf.layers.conv2d(inputs, filters, (1, 1), (1, 1), padding='same', use_bias=False,
							 kernel_initializer=CONV_KERNEL_INITIALIZER, name=prefix + 'expand_conv')
		x = tf.layers.batch_normalization(x, training=training, name=prefix + 'expand_bn')
		x = tf.nn.relu(x, name=prefix+'expand_swish')
	else:
		x = inputs

	# DepthWise Convolution
	x = tf.keras.layers.DepthwiseConv2D(block_args.kernel_size, strides=block_args.strides, depth_multiplier=1,
										padding='same', use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER,
										name=prefix + 'dwconv')(x)
	x = tf.layers.batch_normalization(x, training=training, name=prefix + 'bn')
	x = tf.nn.relu(x, name=prefix+'swish')

	# Squeeze and Excitation phase
	if has_se:
		num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))

		x = Squeeze_excitation_layer(x, filters, num_reduced_filters, name=prefix + 'se_layer')

	# Output phase
	x = tf.layers.conv2d(x, block_args.output_filters, (1, 1), (1, 1), padding='same', use_bias=False,
						 kernel_initializer=CONV_KERNEL_INITIALIZER, name=prefix + 'project_conv')
	x = tf.layers.batch_normalization(x, training=training, name=prefix + 'project_bn')
  
	if block_args.id_skip and all(
			s == 1 for s in block_args.strides
	) and block_args.input_filters == block_args.output_filters:
		if drop_rate and (drop_rate > 0) and keep_dropout_backbone:

			x = tf.nn.dropout(x, keep_prob=1-drop_rate, name=prefix+'dropout')

		x = tf.add(x, inputs, name=prefix + 'add')

	return x


def EfficientNet(img_input, input_shape=None, width_coefficient=1.0, depth_coefficient=1.0, default_resolution=224,
				 dropout_rate=0.2, drop_connect_rate=0.2, depth_divisor=8, blocks_args=DEFAULT_BLOCKS_ARGS,
				 include_top=True, pooling=None, classes=1000, training=True, scope=None, keep_dropout_backbone=True, reuse=None):
	with tf.variable_scope(scope, reuse=reuse):

		# Build stem
		x = img_input
		x = tf.layers.conv2d(x, round_filters(32, width_coefficient, depth_divisor), (3, 3), (2, 2), padding='same',
							 kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False,  name='stem_conv')
		x = tf.layers.batch_normalization(x, training=training, name='stem_bn')
		x = tf.nn.relu(x, name='stem_swish')

		#  Build blocks
		out = []
		num_blocks_total = sum(round_repeats(block_args.num_repeat, depth_coefficient) for block_args in blocks_args)

		block_num = 0
		for idx, block_args in enumerate(blocks_args):
			assert block_args.num_repeat > 0
			# Update block input and output filters based on depth multiplier.
			block_args = block_args._replace(
				input_filters=round_filters(block_args.input_filters,
											width_coefficient, depth_divisor),
				output_filters=round_filters(block_args.output_filters,
											 width_coefficient, depth_divisor),
				num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))
			if block_args.strides == [2, 2]:
				out.append(x)
			# The first block needs to take care of stride and filter size increase.
			drop_rate = drop_connect_rate * float(block_num) / num_blocks_total

			x = mb_conv_block(x, block_args, training=training, keep_dropout_backbone=keep_dropout_backbone,
							  drop_rate=drop_rate, prefix='block{}a_'.format(idx + 1))

			block_num += 1

			if block_args.num_repeat > 1:
				#  pylint: disable=protected-access
				block_args = block_args._replace(
					input_filters=block_args.output_filters, strides=[1, 1])
				#  pylint: enable=protected-access
				for bidx in xrange(block_args.num_repeat - 1):
					drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
					block_prefix = 'block{}{}_'.format(
						idx + 1,
						string.ascii_lowercase[bidx + 1]
					)
					x = mb_conv_block(x, block_args, training=training, keep_dropout_backbone=keep_dropout_backbone,
									  drop_rate=drop_rate, prefix=block_prefix)
					block_num += 1
	out.append(x)
	return out


def EfficientNetB0(Image, input_shape=None, include_top=True, pooling=None, training=True,
				   classes=1000, model_name=None, keep_dropout_backbone=True, reuse=None):
	return EfficientNet(Image, input_shape=input_shape,
						width_coefficient=1.0,
						depth_coefficient=1.0,
						default_resolution=224,
						dropout_rate=0.2,
						include_top=include_top,
						pooling=pooling,
						classes=classes,
						scope=model_name,
						reuse=reuse,
						keep_dropout_backbone=keep_dropout_backbone,
						training=training)


def EfficientNetB1(Image, input_shape=None, include_top=True, pooling=None, training=True,
				   classes=1000, model_name=None, keep_dropout_backbone=True, reuse=None):
	return EfficientNet(Image, input_shape=input_shape,
						width_coefficient=1.0,
						depth_coefficient=1.1,
						default_resolution=240,
						dropout_rate=0.2,
						include_top=include_top,
						pooling=pooling,
						classes=classes,
						scope=model_name,
						reuse=reuse,
						keep_dropout_backbone=keep_dropout_backbone,
						training=training)


def EfficientNetB2(Image, input_shape=None, include_top=True, pooling=None, training=True,
				   classes=1000, model_name=None, keep_dropout_backbone=True, reuse=None):
	return EfficientNet(Image, input_shape=input_shape,
						width_coefficient=1.1,
						depth_coefficient=1.2,
						default_resolution=260,
						dropout_rate=0.3,
						include_top=include_top,
						pooling=pooling,
						classes=classes,
						scope=model_name,
						reuse=reuse,
						keep_dropout_backbone=keep_dropout_backbone,
						training=training)


def EfficientNetB3(Image, input_shape=None, include_top=True, pooling=None, training=True,
				   classes=1000, model_name=None, keep_dropout_backbone=True, reuse=None):
	return EfficientNet(Image, input_shape=input_shape,
						width_coefficient=1.2,
						depth_coefficient=1.4,
						default_resolution=300,
						dropout_rate=0.3,
						include_top=include_top,
						pooling=pooling,
						classes=classes,
						scope=model_name,
						reuse=reuse,
						keep_dropout_backbone=keep_dropout_backbone,
						training=training)


def EfficientNetB4(Image, input_shape=None, include_top=True, pooling=None, training=True,
				   classes=1000, model_name=None, keep_dropout_backbone=True, reuse=None):
	return EfficientNet(Image, input_shape=input_shape,
						width_coefficient=1.4,
						depth_coefficient=1.8,
						default_resolution=380,
						dropout_rate=0.4,
						include_top=include_top,
						pooling=pooling,
						classes=classes,
						scope=model_name,
						reuse=reuse,
						keep_dropout_backbone=keep_dropout_backbone,
						training=training)


def EfficientNetB5(Image, input_shape=None, include_top=True, pooling=None, training=True,
				   classes=1000, model_name=None, keep_dropout_backbone=True, reuse=None):
	return EfficientNet(Image, input_shape=input_shape,
						width_coefficient=1.6,
						depth_coefficient=2.2,
						default_resolution=456,
						dropout_rate=0.4,
						include_top=include_top,
						pooling=pooling,
						classes=classes,
						scope=model_name,
						reuse=reuse,
						keep_dropout_backbone=keep_dropout_backbone,
						training=training)


def EfficientNetB6(Image, input_shape=None, include_top=True, pooling=None, training=True,
				   classes=1000, model_name=None, keep_dropout_backbone=True, reuse=None):
	return EfficientNet(Image, input_shape=input_shape,
						width_coefficient=1.8,
						depth_coefficient=2.6,
						default_resolution=528,
						dropout_rate=0.5,
						include_top=include_top,
						pooling=pooling,
						classes=classes,
						scope=model_name,
						reuse=reuse,
						keep_dropout_backbone=keep_dropout_backbone,
						training=training)


def EfficientNetB7(Image, input_shape=None, include_top=True, pooling=None, training=True,
				   classes=1000, model_name=None, keep_dropout_backbone=True, reuse=None):
	return EfficientNet(Image, input_shape=input_shape,
						width_coefficient=2.0,
						depth_coefficient=3.1,
						default_resolution=600,
						dropout_rate=0.5,
						include_top=include_top,
						pooling=pooling,
						classes=classes,
						scope=model_name,
						reuse=reuse,
						keep_dropout_backbone=keep_dropout_backbone,
						training=training)
