import tensorflow as tf


def decision_head(x, y, class_num, scope, keep_dropout_head, training=True, reuse=None, drop_rate=0.2):

	with tf.variable_scope(scope, reuse=reuse):

		x = tf.concat([x, y], axis=-1)
		x = tf.layers.conv2d(x, 16, (3, 3), strides=(2, 2), padding='same', name='dec_conv2d1')
		x = tf.layers.batch_normalization(x, training=training, name='dec_bn1')
		x = tf.nn.relu(x, name='dec_relu1')
		x = tf.layers.conv2d(x, 16, (3, 3), strides=(1, 1), padding='same', name='dec_conv2d2')
		x = tf.layers.batch_normalization(x, training=training, name='dec_bn2')
		x = tf.nn.relu(x, name='dec_relu2')

		x = tf.layers.conv2d(x, 32, (3, 3), strides=(2, 2), padding='same', name='dec_conv2d3')
		x = tf.layers.batch_normalization(x, training=training, name='dec_bn3')
		x = tf.nn.relu(x, name='dec_relu3')
		x = tf.layers.conv2d(x, 32, (3, 3), strides=(1, 1), padding='same', name='dec_conv2d4')
		x = tf.layers.batch_normalization(x, training=training, name='dec_bn4')
		x = tf.nn.relu(x, name='dec_relu4')

		x = tf.layers.conv2d(x, 32, (3, 3), strides=(2, 2), padding='same', name='dec_conv2d5')

		# de_glob_ds = tf.keras.layers.DepthwiseConv2D(filters=64, kernel_size=(x.shape[1], x.shape[2]),
		#                                              strides=(1, 1), name='GlobalDwConv')(x)

		de_max_po = tf.layers.max_pooling2d(x, pool_size=(x.shape[1], x.shape[2]),
											strides=(1, 1), name='GlobalMaxPooling1')
		de_avg_po = tf.layers.average_pooling2d(x, pool_size=(x.shape[1], x.shape[2]),
												strides=(1, 1), name='GlobalAveragePooling1')
		seg_max_po = tf.layers.max_pooling2d(y, pool_size=(y.shape[1], y.shape[2]),
											 strides=(1, 1), name='GlobalMaxPooling2')
		seg_avg_po = tf.layers.average_pooling2d(y, pool_size=(y.shape[1], y.shape[2]),
												 strides=(1, 1), name='GlobalAveragePooling2')
		# de_glob_ds = tf.layers.Flatten(name='dec_flatten0')(de_glob_ds)
		de_max_po = tf.layers.Flatten(name='dec_flatten1')(de_max_po)
		de_avg_po = tf.layers.Flatten(name='dec_flatten2')(de_avg_po)
		seg_max_po = tf.layers.Flatten(name='dec_flatten3')(seg_max_po)
		seg_avg_po = tf.layers.Flatten(name='dec_faltten4')(seg_avg_po)

		x = tf.concat([de_max_po, de_avg_po, seg_max_po, seg_avg_po], axis=-1)
		if keep_dropout_head:
			x = tf.nn.dropout(x, keep_prob=1-drop_rate)
		x = tf.layers.dense(x, class_num)
		# x = tf.keras.layers.Dense(class_num, use_bias=False)(x)

		return x