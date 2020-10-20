import tensorflow as tf


def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):

	if (conv_type == 'ds'):
		x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides=strides,
											kernel_initializer='he_uniform', data_format='channels_first')(inputs)
	else:
		x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides=strides,
								   kernel_initializer='he_uniform', data_format='channels_first')(inputs)
	x = tf.keras.layers.BatchNormalization(axis=1)(x, training=False)

	if (relu):
		x = tf.keras.layers.Activation('relu')(x)

	return x


def _res_bottleneck(inputs, filters, kernel, t, s, r=False):

	tchannel = tf.keras.backend.int_shape(inputs)[1] * t
	x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))
	x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), padding='same', depth_multiplier=1, data_format='channels_first')(x)
	x = tf.keras.layers.BatchNormalization(axis=1)(x, training=False)
	x = tf.keras.layers.Activation('relu')(x)
	x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)
	if r:
		x = tf.keras.layers.add([x, inputs])

	return x


def bottleneck_block(inputs, filters, kernel, t, strides, n):

	x = _res_bottleneck(inputs, filters, kernel, t, strides)
	for i in range(1, n):
		x = _res_bottleneck(x, filters, kernel, t, 1, True)

	return x


def pyramid_pooling_block(input_tensor, bin_sizes):

	concat_list = [input_tensor]
	w = input_tensor.shape[1]
	h = input_tensor.shape[2]
	for bin_size in bin_sizes:
		x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size), name='ppm_avg_pool_{}'.format(bin_size), data_format='channels_first')(input_tensor)
		x = tf.keras.layers.Conv2D(32, 1, 1, padding='same', name='ppm_con2d_{}'.format(bin_size), data_format='channels_first')(x)
		x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w, h), align_corners=True), name='ppm_lambda{}'.format(bin_size), )(x)
		concat_list.append(x)
	x = tf.keras.layers.concatenate(concat_list, name='ppm_concate')
	x = tf.keras.layers.Conv2D(128, 1, 1, padding='same', name='ppm_con2d', data_format='channels_first')(x)

	return x


def build_fast_scnn(input_tensor, scope, is_training, keep_dropout_backbone):

	with tf.variable_scope(scope):

		lds_layer = conv_block(input_tensor, 'conv', 32, (3, 3), strides=(2, 2))
		lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides=(2, 2))
		lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides=(2, 2))

		gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
		gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
		gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
		gfe_layer = pyramid_pooling_block(gfe_layer, [1, 2, 3, 6])

		ff_layer1 = conv_block(lds_layer, 'conv', 128, (1, 1), padding='same', strides=(1, 1), relu=False)
		ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
		ff_layer2 = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same', data_format='channels_first')(ff_layer2)
		ff_layer2 = tf.keras.layers.BatchNormalization(axis=1)(ff_layer2, training=False)
		ff_layer2 = tf.keras.layers.Activation('relu')(ff_layer2)
		ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None, data_format='channels_first')(ff_layer2)

		ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
		ff_final = tf.keras.layers.BatchNormalization(axis=1)(ff_final, training=False)
		ff_final = tf.keras.layers.Activation('relu')(ff_final)

		classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1),
													 name='DSConv1_classifier', data_format='channels_first')(ff_final)
		classifier = tf.keras.layers.BatchNormalization(axis=1)(classifier, training=False)
		classifier = tf.keras.layers.Activation('relu')(classifier)

		classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1),
													 name='DSConv2_classifier', data_format='channels_first')(classifier)
		classifier = tf.keras.layers.BatchNormalization(axis=1)(classifier, training=False)
		classifier = tf.keras.layers.Activation('relu')(classifier)

		seg_fea = conv_block(classifier, 'conv', 1, (1, 1), strides=(1, 1), padding='same', relu=False)
		if keep_dropout_backbone:
			seg_fea = tf.keras.layers.Dropout(0.3)(seg_fea)

	return [classifier, seg_fea]

