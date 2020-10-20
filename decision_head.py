import tensorflow as tf
from convblock import ConvBatchNormRelu as CBR
from tensorflow.python.ops import math_ops
slim=tf.contrib.slim


def decision_head(x, y, class_num, scope, keep_dropout_head, training, data_format, momentum, mode, reuse=None, drop_rate=0.2, activation='relu'):

	with tf.variable_scope(scope, reuse=reuse):

		channel_axis = 1 if data_format == 'channels_first' else -1

		x = tf.concat([x, y], axis=channel_axis)
		x = CBR(x, 16, 3, 2, training, momentum, mode, name='CBR1', padding='same', data_format=data_format, activation=activation, bn=True)
		x = CBR(x, 16, 3, 1, training, momentum, mode, name='CBR2', padding='same', data_format=data_format, activation=activation, bn=True)
		x = CBR(x, 32, 3, 2, training, momentum, mode, name='CBR3', padding='same', data_format=data_format, activation=activation, bn=True)
		x = CBR(x, 32, 3, 1, training, momentum, mode, name='CBR4', padding='same', data_format=data_format, activation=activation, bn=True)
		x = CBR(x, 32, 3, 2, training, momentum, mode, name='CBR5', padding='same', data_format=data_format, activation=None, bn=False)

		# de_glob_ds = tf.keras.layers.DepthwiseConv2D(filters=64, kernel_size=(x.shape[1], x.shape[2]),
		#                                              strides=(1, 1), name='GlobalDwConv')(x)

		reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]

		vector1 = math_ops.reduce_mean(x, reduction_indices, name='pool4', keepdims=True)
		vector2 = math_ops.reduce_max(x, reduction_indices, name='pool5', keepdims=True)
		vector3 = math_ops.reduce_mean(y, reduction_indices, name='pool6', keepdims=True)
		vector4 = math_ops.reduce_max(y, reduction_indices, name='pool7', keepdims=True)

		# de_glob_ds = tf.layers.Flatten(name='dec_flatten0')(de_glob_ds)
		vector = tf.concat([vector1, vector2, vector3, vector4], axis=channel_axis)
		vector = tf.squeeze(vector, axis=reduction_indices)

		if keep_dropout_head:
			vector = tf.nn.dropout(vector, keep_prob=1-drop_rate)
		logits = slim.fully_connected(vector, class_num, activation_fn=None)

		return logits