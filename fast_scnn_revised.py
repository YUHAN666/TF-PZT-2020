from config import CLASS_NUM, BIN_SIZE, IMAGE_SIZE, DATA_FORMAT
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
import os
import cv2
from tqdm import tqdm
slim=tf.contrib.slim


kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)


def SELayer(x, out_dim, se_dim, data_format, name):
	with tf.variable_scope(name):
		axis = [1, 2] if data_format == "channels_last" else [2, 3]

		squeeze = tf.reduce_mean(x, axis=axis, keepdims=True, name=name + "_gap")

		excitation = tf.layers.Conv2D(se_dim, 1, strides=1, data_format=data_format,
									  kernel_initializer=kernel_initializer, padding='same', name=name + "_fc1")(
			squeeze)

		excitation = tf.nn.relu(excitation, name=name + '_relu')

		excitation = tf.layers.Conv2D(out_dim, 1, strides=1, data_format=data_format,
									  kernel_initializer=kernel_initializer, padding='same', name=name + "_fc2")(
			excitation)
		excitation = tf.nn.sigmoid(excitation,name = name+'_sigmoid')
		# excitation = tf.clip_by_value(excitation, 0, 1, name=name + '_hsigmoid')

		scale = x * excitation

		return scale


def conv_block(inputs, conv_type, filters, kernel_size, strides, training, bn_momentum, name=None, padding='same',
			   data_format='channels_first', relu=True, use_bias=False):
	if (conv_type == 'ds'):
		x = tf.layers.separable_conv2d(inputs, filters, kernel_size, strides, name=name + '_ds_conv', padding=padding,
									   data_format=data_format, use_bias=use_bias)
	else:
		x = tf.layers.conv2d(inputs, filters, kernel_size, strides, name=name + '_conv2d', padding='same',
							 data_format=data_format, use_bias=use_bias)

	axis = 1 if data_format == 'channels_first' else -1

	x = tf.layers.batch_normalization(x, axis=axis, training=training, momentum=bn_momentum, name=name + '_bn')

	if (relu):
		x = tf.nn.relu(x, name=name + '_relu')
	return x


def _res_bottleneck(inputs, filters, kernel, t, s, training, bn_momentum, name=None, drop_rate=0.5,
					data_format='channels_first', res=False):
	channel_axis = 1 if data_format == 'channels_first' else -1

	nIn = inputs.get_shape().as_list()[channel_axis]
	tchannel = nIn * t

	# expand
	x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1), name=name + '_conv',
				   training=training, bn_momentum=bn_momentum, data_format=data_format)
	# dw_conv
	x = tf.keras.layers.DepthwiseConv2D(kernel, s, depth_multiplier=1, data_format=data_format, padding='same')(x)

	x = tf.layers.batch_normalization(x, axis=channel_axis, training=training, momentum=bn_momentum, name=name + '_bn')
	x = tf.nn.relu(x, name=name + 'relu')

	# SE
	se_dim = max(1, nIn // 4)
	x = SELayer(x, tchannel, se_dim, data_format, name=name + '_se')

	# project
	x = conv_block(x, 'conv', filters, (1, 1), strides=1, training=training, bn_momentum=bn_momentum,
				   name=name + '_conv2',
				   padding='same', data_format=data_format, relu=False, use_bias=False)

	if res:
		# drop_out
		if channel_axis == -1:
			x = tf.layers.dropout(x, drop_rate, training=training, noise_shape=(None, 1, 1, None), name='dropout')
		else:
			x = tf.layers.dropout(x, drop_rate, training=training, noise_shape=(None, None, 1, 1), name='dropout')
		x = x + inputs
	return x


def bottleneck_block(inputs, filters, kernel, t, strides, n, training, bn_momentum, drop_rate,
					 data_format='channels_first', name=None):
	x = _res_bottleneck(inputs, filters, kernel, t, strides, training, bn_momentum=bn_momentum, drop_rate=drop_rate,
						data_format=data_format, name=name + '_res')
	for i in range(1, n):
		x = _res_bottleneck(x, filters, kernel, t, 1, training, bn_momentum=bn_momentum, drop_rate=drop_rate,
							data_format=data_format, name=name + str(i) + '_res', res=True)
	return x


def pyramid_pooling_block(input_tensor, nOut, bin_sizes, training, bn_momentum, data_format='channels_first',
						  name=None):
	concat_list = [input_tensor]

	if data_format == 'channels_last':
		w = input_tensor.get_shape().as_list()[2]
		h = input_tensor.get_shape().as_list()[1]
		axis = -1
	else:
		w = input_tensor.get_shape().as_list()[3]
		h = input_tensor.get_shape().as_list()[2]
		axis = 1

	nbin = len(bin_sizes)
	laynOut = nOut // nbin
	outlist = nbin * [laynOut]
	outlist[0] = outlist[0] + (nOut - laynOut * nbin)

	n = 0
	for bin_size in bin_sizes:
		n = n + 1
		x = tf.layers.average_pooling2d(input_tensor, pool_size=(
		h - (bin_size - 1) * (h // bin_size), w - (bin_size - 1) * (w // bin_size)),
										strides=(h // bin_size, w // bin_size),
										data_format=data_format, name=name + '_' + str(n) + '_agp2d')
		x = conv_block(x, 'conv', outlist[n - 1], (1, 1), strides=(1, 1), padding='valid',
					   name=name + '_' + str(n) + 'conv',
					   training=training, bn_momentum=bn_momentum, data_format=data_format)

		if data_format == 'channels_last':
			x = tf.image.resize_images(x, (h, w), align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		else:
			x = tf.transpose(x, [0, 2, 3, 1])  # NCHW->NHWC
			x = tf.image.resize_images(x, (h, w), align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			x = tf.transpose(x, [0, 3, 1, 2])  # NHWC -> NCHW
		concat_list.append(x)

	x = tf.concat(concat_list, axis=axis)
	x = conv_block(x, 'conv', nOut, (1, 1), strides=(1, 1), training=training, bn_momentum=bn_momentum,
				   name=name + 'conv', padding='valid', data_format=data_format)
	return x


def LearningToDownsample(inputs, training, bn_momentum, data_format='channels_first', scope='lds', reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		lds_layer = conv_block(inputs, 'conv', 32, (3, 3), strides=(2, 2), training=training,
							   bn_momentum=bn_momentum, data_format=data_format, name='conv1', padding='valid')
		lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides=(2, 2), training=training,
							   bn_momentum=bn_momentum, data_format=data_format, name='dsconv2')
		lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides=(2, 2), training=training,
							   bn_momentum=bn_momentum, data_format=data_format, name='dsconv3')
		return lds_layer


def GlobalFeatureExtractor(inputs, training, bn_momentum, drop_rate, data_format='channels_first', scope='gfe',
						   reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		x = bottleneck_block(inputs, 64, (3, 3), 6, (2, 2), n=3,
							 training=training, bn_momentum=bn_momentum, drop_rate=drop_rate, data_format=data_format,
							 name='btnk1')
		x = bottleneck_block(x, 96, (3, 3), 6, (2, 2), n=3,
							 training=training, bn_momentum=bn_momentum, drop_rate=drop_rate, data_format=data_format,
							 name='btnk2')
		x = bottleneck_block(x, 128, (3, 3), 6, (1, 1), n=3,
							 training=training, bn_momentum=bn_momentum, drop_rate=drop_rate, data_format=data_format,
							 name='btnk3')
		x = pyramid_pooling_block(x, 128, BIN_SIZE,
								  training=training, bn_momentum=bn_momentum, data_format=data_format, name='ppb')
		return x


def FeatureFusion(higher_res_feature, low_res_feature, training, bn_momentum, data_format='channels_first', scope='ff',
				  reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		channel_axis = 1 if data_format == 'channels_first' else -1

		ff_layer1 = conv_block(higher_res_feature, 'conv', 128, (1, 1), strides=(1, 1),
							   training=training, bn_momentum=bn_momentum, data_format=data_format, name='conv',
							   padding='same', relu=False)

		ff_layer2 = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1,
													padding='same', data_format=data_format)(low_res_feature)
		ff_layer2 = tf.layers.batch_normalization(ff_layer2, axis=channel_axis, training=training, momentum=bn_momentum,
												  name='bn')

		ff_layer2 = tf.nn.relu(ff_layer2, name='relu1')
		ff_layer2 = tf.layers.conv2d(ff_layer2, 128, (1, 1), (1, 1), padding='same', data_format=data_format,
									 name='conv2d')
		ff_layer2 = tf.keras.layers.UpSampling2D((4, 4), data_format=data_format)(ff_layer2)

		x = ff_layer1 + ff_layer2
		x = tf.nn.relu(x, name='relu2')
		return x


def Classifier(inputs, training, bn_momentum, drop_rate, data_format='channels_first', scope='classifier', reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		x = conv_block(inputs, 'ds', 128, (3, 3), strides=(1, 1), training=training, bn_momentum=bn_momentum,
					   data_format=data_format, name='dsconv1', padding='same')
		features = conv_block(x, 'ds', 128, (3, 3), strides=(1, 1), training=training, bn_momentum=bn_momentum,
							  data_format=data_format, name='dsconv2', padding='same')
		# features = tf.nn.dropout(features,keep_prob,name = 'dropout')

		features = tf.layers.dropout(features, drop_rate, training=training, name='dropout')

		logits = conv_block(features, 'conv', CLASS_NUM, (1, 1), strides=(1, 1), training=training,
							bn_momentum=bn_momentum,
							data_format=data_format, name='conv3', padding='same', relu=False)
		# logits = tf.keras.layers.UpSampling2D((8, 8),data_format=data_format)(logits)
		return features, logits


def SegmentNet(inputs, scope, is_training, bn_momentum, drop_rate, data_format='channels_first', reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		# if data_format == 'channels_first':
		# 	inputs = tf.transpose(inputs, [0, 3, 1, 2])  # NHWC -> NCHW

		lds = LearningToDownsample(inputs, is_training, bn_momentum,
								   data_format=data_format, scope='lds', reuse=reuse)
		gfe = GlobalFeatureExtractor(lds, is_training, bn_momentum, drop_rate,
									 data_format=data_format, scope='gfe', reuse=reuse)
		ff = FeatureFusion(lds, gfe, is_training, bn_momentum,
						   data_format=data_format, scope='ff', reuse=reuse)
		features, logits = Classifier(ff, is_training, bn_momentum, drop_rate,
									  data_format=data_format, scope='classify', reuse=reuse)

		logits = tf.keras.layers.UpSampling2D((8, 8), data_format=data_format)(logits)
		mask = tf.sigmoid(logits, name='sigmoid')
		return features, logits, mask


def DecisionNet(feature, mask, scope, is_training, bn_momentum=0.99, num_classes=1, data_format='channels_first',
				reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		channel_axis = 1 if data_format == 'channels_first' else -1

		mask = tf.layers.max_pooling2d(mask, (8, 8), (8, 8), data_format=data_format, padding='same')

		net = tf.concat([feature, mask], axis=channel_axis)
		net = conv_block(net, 'ds', 8, (3, 3), strides=(2, 2), training=is_training, bn_momentum=bn_momentum,
						 data_format=data_format, name='dcn_dscon1', padding='same')
		net = conv_block(net, 'ds', 16, (3, 3), strides=(2, 2), training=is_training, bn_momentum=bn_momentum,
						 data_format=data_format, name='dcn_scon2', padding='same')
		net = conv_block(net, 'conv', 32, (3, 3), strides=(2, 2), training=is_training, bn_momentum=bn_momentum,
						 data_format=data_format, name='dcn_dscon3', padding='same')

		reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]

		vector1 = math_ops.reduce_mean(net, reduction_indices, name='pool4', keepdims=True)
		vector2 = math_ops.reduce_max(net, reduction_indices, name='pool5', keepdims=True)
		vector3 = math_ops.reduce_mean(mask, reduction_indices, name='pool6', keepdims=True)
		vector4 = math_ops.reduce_max(mask, reduction_indices, name='pool7', keepdims=True)

		vector = tf.concat([vector1, vector2, vector3, vector4], axis=channel_axis)
		vector = tf.squeeze(vector, axis=reduction_indices)
		logits = slim.fully_connected(vector, num_classes, activation_fn=None)

		output = tf.argmax(logits, axis=1, name='argmax')
		return logits, output



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



"""-----------------------------------------train  segmentation-----------------------------------------------"""

sess = tf.Session()
checkPoint_dir = "checkpoint"
with sess.as_default():
	image_input = tf.placeholder(tf.float32, shape=(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1]), name='Image')
	label = tf.placeholder(tf.float32, shape=(1, 1), name='Label')
	Mask = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), name='mask')
	is_training_seg = tf.placeholder(tf.bool, name='is_training_seg')
	is_training_dec = tf.placeholder(tf.bool, name='is_training_dec')

	features, logits_pixel, mask = SegmentNet(image_input, 'segment', is_training_seg, drop_rate=0.5,
											  data_format=DATA_FORMAT, bn_momentum=0.9)

	logits_class, output_class = DecisionNet(features, mask, 'decision', is_training_dec,
											 data_format=DATA_FORMAT, bn_momentum=0.9)

	decision_out = tf.nn.sigmoid(logits_class, name='decision_out')

	#        logits_pixel=tf.reshape(logits_pixel,[self.__batch_size,-1])
	#        PixelLabel_reshape=tf.reshape(PixelLabel,[self.__batch_size,-1])

	if DATA_FORMAT == 'channels_first':
		logits_pixel = tf.transpose(logits_pixel, [0, 2, 3, 1])  # NCHW->NHWC
	loss_pixel = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=Mask))
	loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_class, labels=label))
	loss_total = loss_pixel + loss_class

	optimizer = tf.train.AdamOptimizer(0.001)
	train_var_list = [v for v in tf.trainable_variables()]
	train_segment_var_list = [v for v in tf.trainable_variables() if 'segment' in v.name]
	train_decision_var_list = [v for v in tf.trainable_variables() if 'decision' in v.name]

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	updata_ops_segment = [v for v in update_ops if 'segment' in v.name]
	updata_ops_decision = [v for v in update_ops if 'decision' in v.name]

	with tf.control_dependencies(updata_ops_segment):
		optimize_segment = optimizer.minimize(loss_pixel, var_list=train_segment_var_list)

	with tf.control_dependencies(updata_ops_decision):
		optimize_decision = optimizer.minimize(loss_class, var_list=train_decision_var_list)

	with tf.control_dependencies(update_ops):
		optimize_total = optimizer.minimize(loss_total, var_list=train_var_list)

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


with sess.as_default():
	# sess.run(tf.global_variables_initializer())
	seg_epochs = 15
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
												 loss_pixel],
												feed_dict={image_input: img_batch,
														   Mask: mask_batch,
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

			total_loss_value_batch = sess.run(loss_pixel,
											  feed_dict={image_input: img_batch,
														 Mask: mask_batch,
														 label: label_batch,
														 is_training_seg: False,
														 is_training_dec: False})
			num_step = num_step + 1
			val_loss += total_loss_value_batch
		val_loss /= num_step

		print('train_loss:{},   val_loss:{}'.format(iter_loss, val_loss))
		saver.save(sess, os.path.join(checkPoint_dir, 'ckp'), global_step=seg_epochs)

"""-----------------------------------------train  decision----------------------------------------------------------"""
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
															  loss_class,
															  decision_out],
												   feed_dict={image_input: img_batch,
															  Mask: mask_batch,
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

				total_loss_value_batch, decision = sess.run([loss_class, decision_out],
												  feed_dict={image_input: img_batch,
															 Mask: mask_batch,
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
# """--------------------------------------Save Pb from checkpoint------------------------------------------"""
# checkPoint_dir = 'checkpoint'
# session = tf.Session()
# with session.as_default():
#
# 	image_input = tf.placeholder(tf.float32, shape=(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1]), name='Image')
# 	features, logits_pixel, mask = SegmentNet(image_input, 'segment', False, drop_rate=0,
# 											  data_format=DATA_FORMAT, bn_momentum=0.9)
#
# 	logits_class, output_class = DecisionNet(features, mask, 'decision', False,
# 											 data_format=DATA_FORMAT, bn_momentum=0.9)
#
# 	decision_out = tf.nn.sigmoid(logits_class, name='decision_out')
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
# 	pbpath = os.path.join(__pb_model_path, 'tensorrt_fastscnn.pb')
# 	with tf.gfile.GFile(pbpath, mode='wb') as f:
# 		f.write(output_graph_def.SerializeToString())
#
#
# pb_Model_dir = "pbMode"
# save_PbModel(session, pb_Model_dir)
