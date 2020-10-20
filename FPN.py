""" Implementation of BiFPN, FPN, PAN """

import tensorflow as tf
from tensorflow import keras as keras
from config import IMAGE_SIZE, ACTIVATION
from mish import mish
from swish import swish
from convblock import ConvBatchNormRelu as CBR


def DepthwiseConvBlock(input_tensor, kernel_size, strides, name, data_format, is_training=False):
	if data_format == 'channels_first':
		axis = 1
	else:
		axis = -1

	x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
								use_bias=False, name='{}_dconv'.format(name), data_format=data_format)(input_tensor)

	x = tf.layers.batch_normalization(x, training=is_training, name='{}_bn'.format(name), axis=axis)
	if ACTIVATION == 'swish':
		x = swish(x, name='{}_swish'.format(name))
	elif ACTIVATION == 'mish':
		x = mish(x)
	else:
		x = keras.layers.ReLU(name='{}_relu'.format(name))(x)
	return x


# def ConvBlock(input_tensor, num_channels, kernel_size, strides, name, is_training=False):
# 	x = keras.layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
# 					   use_bias=False, name='{}_conv'.format(name))(input_tensor)
# 	x = tf.layers.batch_normalization(x, training=is_training, name='{}_bn'.format(name))
# 	if ACTIVATION == 'swish':
# 		x = swish(x, name='{}_swish'.format(name))
# 	elif ACTIVATION == 'mish':
# 		x = mish(x)
# 	else:
# 		x = keras.layers.ReLU(name='{}_relu'.format(name))(x)
# 	return x


def bifpn_neck(features, num_channels, scope, momentum, mode, data_format, is_training=False, reuse=None):
	""" BiFPN """

	with tf.variable_scope(scope, reuse=reuse):
		P3_in, P4_in, P5_in, P6_in, P7_in = features
		# P3_in = ConvBlock(P3_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P3')
		# P4_in = ConvBlock(P4_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P4')
		# P5_in = ConvBlock(P5_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P5')
		# P6_in = ConvBlock(P6_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P6')
		# P7_in = ConvBlock(P7_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BiFPN_P7')

		P3_in = CBR(P3_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P3',
					padding='same', data_format=data_format, activation=ACTIVATION, bn=True)
		P4_in = CBR(P4_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P4',
					padding='same', data_format=data_format, activation=ACTIVATION, bn=True)
		P5_in = CBR(P5_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P5',
					padding='same', data_format=data_format, activation=ACTIVATION, bn=True)
		P6_in = CBR(P6_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P6',
					padding='same', data_format=data_format, activation=ACTIVATION, bn=True)
		P7_in = CBR(P7_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P7',
					padding='same', data_format=data_format, activation=ACTIVATION, bn=True)
		# upsample
		P7_U = keras.layers.UpSampling2D(interpolation='bilinear', data_format=data_format)(P7_in)
		# P7_U = tf.image.resize_images(P7_in, (IMAGE_SIZE[0]//16, IMAGE_SIZE[1]//16), align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		P6_td = keras.layers.Add()([P7_U, P6_in])
		P6_td = DepthwiseConvBlock(P6_td, kernel_size=3, strides=1, data_format=data_format, is_training=is_training, name='BiFPN_U_P6')
		P6_U = keras.layers.UpSampling2D(interpolation='bilinear', data_format=data_format)(P6_td)
		# P6_U = tf.image.resize_images(P6_td, (IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8), align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		P5_td = keras.layers.Add()([P6_U, P5_in])
		P5_td = DepthwiseConvBlock(P5_td, kernel_size=3, strides=1, data_format=data_format, is_training=is_training, name='BiFPN_U_P5')
		P5_U = keras.layers.UpSampling2D(interpolation='bilinear', data_format=data_format)(P5_td)
		# P5_U = tf.image.resize_images(P5_td, (IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4), align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		P4_td = keras.layers.Add()([P5_U, P4_in])
		P4_td = DepthwiseConvBlock(P4_td, kernel_size=3, strides=1, data_format=data_format, is_training=is_training, name='BiFPN_U_P4')
		P4_U = keras.layers.UpSampling2D(interpolation='bilinear', data_format=data_format)(P4_td)
		# P4_U = tf.image.resize_images(P4_td, (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		P3_out = keras.layers.Add()([P4_U, P3_in])
		P3_out = DepthwiseConvBlock(P3_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training, name='BiFPN_U_P3')
		# downsample
		P3_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format=data_format)(P3_out)
		P4_out = keras.layers.Add()([P3_D, P4_td, P4_in])
		P4_out = DepthwiseConvBlock(P4_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training, name='BiFPN_D_P4')
		P4_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format=data_format)(P4_out)
		P5_out = keras.layers.Add()([P4_D, P5_td, P5_in])
		P5_out = DepthwiseConvBlock(P5_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training, name='BiFPN_D_P5')
		P5_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format=data_format)(P5_out)
		P6_out = keras.layers.Add()([P5_D, P6_td, P6_in])
		P6_out = DepthwiseConvBlock(P6_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training, name='BiFPN_D_P6')
		P6_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format=data_format)(P6_out)
		P7_out = keras.layers.Add()([P6_D, P7_in])
		P7_out = DepthwiseConvBlock(P7_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training, name='BiFPN_D_P7')

	return P3_out, P4_out, P5_out, P6_out, P7_out



def fpn_neck(feature, class_num, drop_rate, keep_dropout_head,
                          scope, TOP_DOWN_PYRAMID_SIZE=128, training=True, reuse=None):

	C5 = feature[4]
	C4 = feature[3]
	C3 = feature[2]
	C2 = feature[1]
	# C1 = feature[0]
	with tf.variable_scope(scope, reuse=reuse):

		P5 = tf.layers.conv2d(C5, TOP_DOWN_PYRAMID_SIZE, (1, 1),   name='c5_conv')
		P4 = keras.layers.UpSampling2D()(P5)
		P4 = tf.add(P4, tf.layers.conv2d(C4, TOP_DOWN_PYRAMID_SIZE, (1, 1),   name='c4_conv'))
		P3 = keras.layers.UpSampling2D()(P4)
		P3 = tf.add(P3, tf.layers.conv2d(C3, TOP_DOWN_PYRAMID_SIZE, (1, 1),   name='c3_conv'))
		P2 = keras.layers.UpSampling2D()(P3)
		P2 = tf.add(P2, tf.layers.conv2d(C2, TOP_DOWN_PYRAMID_SIZE, (1, 1),   name='c2_conv'))
		# P1 = tf.add(tf.image.resize_images(P2, (IMAGE_HEIGHT//2, IMAGE_WIDTH//2), method=tf.image.ResizeMethod.BILINEAR),
		#             tf.layers.conv2d(C1, TOP_DOWN_PYRAMID_SIZE, (1, 1),   name='c1_conv'))
		P1 = keras.layers.UpSampling2D()(P2)

		classifier = tf.layers.SeparableConv2D(128, (3, 3), strides=(1, 1),
                                               padding='same', name='classifier_DWConv1')(P1)
		classifier = tf.layers.batch_normalization(classifier, training=training, name='classifier_bn1')
		classifier = tf.nn.relu(classifier, name='classifier_relu1')
		classifier = tf.layers.SeparableConv2D(128, (3, 3), strides=(1, 1),
                                               padding='same', name='classifier_DWConv2')(classifier)
		classifier = tf.layers.batch_normalization(classifier, training=training, name='classifier_bn2')
		classifier = tf.nn.relu(classifier, name='classifier_out')
		if keep_dropout_head:
			segfea = tf.nn.dropout(classifier, keep_prob=0.9, name='segfea_dropout')
		else:
			segfea = classifier
		segfea = tf.layers.conv2d(segfea, class_num, kernel_size=(1, 1), strides=(1, 1), name='segfea')
		segfea = tf.layers.batch_normalization(segfea, training=training, name='segfea_bn')

		return [classifier, segfea]


def PAN(feature, TOP_DOWN_PYRAMID_SIZE=128, training=True):

	C5 = feature[4]
	C4 = feature[3]
	C3 = feature[2]
	C2 = feature[1]

	P5 = tf.layers.conv2d(C5, TOP_DOWN_PYRAMID_SIZE, (1, 1), name='c5_conv')
	P4 = keras.layers.UpSampling2D()(P5)
	P4 = tf.add(P4, tf.layers.conv2d(C4, TOP_DOWN_PYRAMID_SIZE, (1, 1), name='c4_conv'))
	P3 = keras.layers.UpSampling2D()(P4)
	P3 = tf.add(P3, tf.layers.conv2d(C3, TOP_DOWN_PYRAMID_SIZE, (1, 1), name='c3_conv'))
	P2 = keras.layers.UpSampling2D()(P3)
	P2 = tf.add(P2, tf.layers.conv2d(C2, TOP_DOWN_PYRAMID_SIZE, (1, 1), name='c2_conv'))
	# P1 = tf.add(tf.image.resize_images(P2, (IMAGE_HEIGHT//2, IMAGE_WIDTH//2), method=tf.image.ResizeMethod.BILINEAR),
	#             tf.layers.conv2d(C1, TOP_DOWN_PYRAMID_SIZE, (1, 1),   name='c1_conv'))
	# P1 = keras.layers.UpSampling2D()(P2)

	N2 = P2
	N3 = tf.layers.conv2d(N2, TOP_DOWN_PYRAMID_SIZE, (3, 3), (2, 2), padding='same', use_bias=False)
	N3 = tf.layers.batch_normalization(N3, training=training)
	N3 = tf.nn.relu(N3)
	N3 = N3 + P3
	N3 = tf.layers.conv2d(N3, TOP_DOWN_PYRAMID_SIZE, (3, 3), (1, 1), padding='same', use_bias=False)
	N3 = tf.layers.batch_normalization(N3, training=training)
	N3 = tf.nn.relu(N3)

	N4 = tf.layers.conv2d(N3, TOP_DOWN_PYRAMID_SIZE, (3, 3), (2, 2), padding='same', use_bias=False)
	N4 = tf.layers.batch_normalization(N4, training=training)
	N4 = tf.nn.relu(N4)
	N4 = N4 + P4
	N4 = tf.layers.conv2d(N4, TOP_DOWN_PYRAMID_SIZE, (3, 3), (1, 1), padding='same', use_bias=False)
	N4 = tf.layers.batch_normalization(N4, training=training)
	N4 = tf.nn.relu(N4)

	N5 = tf.layers.conv2d(N4, TOP_DOWN_PYRAMID_SIZE, (3, 3), (2, 2), padding='same', use_bias=False)
	N5 = tf.layers.batch_normalization(N5, training=training)
	N5 = tf.nn.relu(N5)
	N5 = N5 + P5
	N5 = tf.layers.conv2d(N5, TOP_DOWN_PYRAMID_SIZE, (3, 3), (1, 1), padding='same', use_bias=False)
	N5 = tf.layers.batch_normalization(N5, training=training)
	N5 = tf.nn.relu(N5)

	return N2, N3, N4, N5


