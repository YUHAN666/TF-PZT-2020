import tensorflow as tf
from tensorflow import keras as keras


def DepthwiseConvBlock(input_tensor, kernel_size, strides, name, is_training=False):
	x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
								use_bias=False, name='{}_dconv'.format(name))(input_tensor)
	x = tf.layers.batch_normalization(x, training=is_training, name='{}_bn'.format(name))
	x = keras.layers.ReLU(name='{}_relu'.format(name))(x)
	return x


def ConvBlock(input_tensor, num_channels, kernel_size, strides, name, is_training=False):
	x = keras.layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
					   use_bias=False, name='{}_conv'.format(name))(input_tensor)
	x = tf.layers.batch_normalization(x, training=is_training, name='{}_bn'.format(name))
	x = keras.layers.ReLU(name='{}_relu'.format(name))(x)
	return x


def refine(input_tensor, inter_channel, scope, is_training):

	with tf.variable_scope(scope):

		g = keras.layers.Conv2D(inter_channel, (1, 1), (1, 1), use_bias=False)(input_tensor)
		g = tf.layers.batch_normalization(g, training=is_training)
		g = tf.reshape(g, (g.shape[0], -1, g.shape[-1]))
		g = tf.transpose(g, (0, 2, 1))
		theta = keras.layers.Conv2D(inter_channel, (1, 1), (1, 1), use_bias=False)(input_tensor)
		theta = tf.layers.batch_normalization(theta, training=is_training)
		theta = tf.reshape(theta, (theta.shape[0], -1, theta.shape[-1]))
		theta = tf.transpose(theta, (0, 2, 1))
		phi = keras.layers.Conv2D(inter_channel, (1, 1), (1, 1), use_bias=False)(input_tensor)
		phi = tf.layers.batch_normalization(phi, training=is_training)
		phi = tf.reshape(phi, (phi.shape[0], -1, phi.shape[-1]))

		pairwise_weight = tf.matmul(theta, phi)
		pairwise_weight = keras.layers.ReLU()(pairwise_weight)

		y = tf.matmul(pairwise_weight, g)
		y = tf.transpose(y, (0, 2, 1))
		y = tf.reshape(y, (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], inter_channel))

		y = keras.layers.Conv2D(int(input_tensor.shape[-1]), (1, 1), (1, 1), use_bias=False)(y)
		y = tf.layers.batch_normalization(y, training=is_training)
		y = keras.layers.Add()([input_tensor, y])

		return y


def bfp_segmentation_head(features, num_channels, scope, is_training):

	with tf.variable_scope(scope):

		P3_in, P4_in, P5_in, P6_in, P7_in = features
		P3_in = ConvBlock(P3_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BFP_P3')
		P4_in = ConvBlock(P4_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BFP_P4')
		P5_in = ConvBlock(P5_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BFP_P5')
		P6_in = ConvBlock(P6_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BFP_P6')
		P7_in = ConvBlock(P7_in, num_channels, kernel_size=1, strides=1, is_training=is_training, name='BFP_P7')

		P3 = keras.layers.MaxPooling2D((4, 4), (4, 4))(P3_in)
		P4 = keras.layers.MaxPooling2D((2, 2), (2, 2))(P4_in)
		P6 = keras.layers.UpSampling2D((2, 2))(P6_in)
		P7 = keras.layers.UpSampling2D((4, 4))(P7_in)

		P = keras.layers.Add()([P3, P4, P5_in, P6, P7])/5

		P = refine(P, 32, scope, is_training)

		P3 = keras.layers.UpSampling2D((4, 4))(P)
		P4 = keras.layers.UpSampling2D((2, 2))(P)
		P5 = P
		P6 = keras.layers.MaxPooling2D((2, 2), (2, 2))(P)
		P7 = keras.layers.MaxPooling2D((4, 4), (4, 4))(P)

		P3_out = keras.layers.Add()([P3, P3_in])
		P4_out = keras.layers.Add()([P4, P4_in])
		P5_out = keras.layers.Add()([P5, P5_in])
		P6_out = keras.layers.Add()([P6, P6_in])
		P7_out = keras.layers.Add()([P7, P7_in])

		return P3_out, P4_out, P5_out, P6_out, P7_out