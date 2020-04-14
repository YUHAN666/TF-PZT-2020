import os
import numpy as np
import tensorflow as tf
from config import CLASS_NUM, IMAGE_SIZE, TRAIN_MODE_IN_TRAIN, TRAIN_MODE_IN_TEST, TRAIN_MODE_IN_VALID, IMAGE_MODE
from efficientnet import EfficientNetB0
from decision_head import decision_head
from BFP import bfp_segmentation_head
from FPN import fpn_segmentation_head, bifpn_segmentation_head
from MixNet import MixNetSmall
from GhostNet import ghostnet_base
from MixNet_official import build_model_base
from fast_scnn import build_fast_scnn
from SiNet import SINet


class Model(object):

	def __init__(self, sess, param):
		self.step = 0
		self.session = sess
		self.__learn_rate = param["learn_rate"]
		self.__learn_rate = param["learn_rate"]
		self.__max_to_keep = param["max_to_keep"]
		self.__checkPoint_dir = param["checkPoint_dir"]
		self.__pb_model_path = param["pb_Mode_dir"]
		self.__restore = param["b_restore"]
		self.__mode = param["mode"]
		self.backbone = param["backbone"]
		self.tensorboard_logdir = param["tensorboard_logdir"]
		self.segmentation_head = param["segmentation_head"]
		if param["mode"] == 'testing' or param["mode"] == 'savePb':
			self.is_training = TRAIN_MODE_IN_TEST
		else:
			self.is_training = TRAIN_MODE_IN_TRAIN
		if param["mode"] == "train_segmentation":
			self.keep_dropout_backbone = True
			self.keep_dropout_head = True
		elif param["mode"] == "train_decision":
			self.keep_dropout_backbone = False
			self.keep_dropout_head = True
		else:
			self.keep_dropout_backbone = False
			self.keep_dropout_head = False
		self.__batch_size = param["batch_size"]
		self.__batch_size_inference = param["batch_size_inference"]

		if param["mode"] == 'savePb' or param["mode"] == 'visualization':
			self.is_bp = True
		else:
			self.is_bp = False

		# Building graph
		with self.session.as_default():
			self.build_model()
		# 参数初始化，或者读入参数
		with self.session.as_default():
			self.init_op.run()
			var_list = tf.trainable_variables()
			g_list = tf.global_variables()
			# for i in g_list:
			#     if 'var' in i.name:
			#         print(i)
			bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
			bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
			var_list += bn_moving_vars
			self.__saver = tf.train.Saver(var_list, max_to_keep=self.__max_to_keep)
			# Loading last save if needed
			if self.__restore:
				ckpt = tf.train.latest_checkpoint(self.__checkPoint_dir)
				if ckpt:
					self.step = int(ckpt.split('-')[1])
					self.__saver.restore(self.session, ckpt)
					print('Restoring from epoch:{}'.format(self.step))
					self.step += 1

	def build_model(self):

		# tf.summary.image('input_image', image_input, 10)
		# tf.summary.image('mask', mask, 10)

		if (self.is_bp == False):
			is_training_seg = tf.placeholder(tf.bool, name='is_training_seg')
			is_training_dec = tf.placeholder(tf.bool, name='is_training_dec')
			if IMAGE_MODE == 0:
				image_input = tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
											 name='Image')
				num_ch = 1
			else:
				image_input = tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
											 name='Image')
				num_ch = 3
			label = tf.placeholder(tf.float32, shape=(self.__batch_size, CLASS_NUM), name='Label')
			mask = tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], CLASS_NUM),
								  name='mask')

		else:
			is_training_seg = False
			is_training_dec = False
			if IMAGE_MODE == 0:
				image_input = tf.placeholder(tf.float32,
											 shape=(self.__batch_size_inference, IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
											 name='Image')
				num_ch = 1
			else:
				image_input = tf.placeholder(tf.float32,
											 shape=(self.__batch_size_inference, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
											 name='Image')
				num_ch = 3
			label = tf.placeholder(tf.float32, shape=(self.__batch_size_inference, CLASS_NUM), name='Label')
			mask = tf.placeholder(tf.float32, shape=(self.__batch_size_inference, IMAGE_SIZE[0], IMAGE_SIZE[1], CLASS_NUM),
								  name='mask')

		# create backbone
		if self.backbone == 'mixnet':
			feature_list = MixNetSmall(image_input, scope='segmentation', include_top=False,
							  	keep_dropout_backbone=self.keep_dropout_backbone, training=is_training_seg)

		elif self.backbone == 'mixnet_official':
			feature_list = build_model_base(image_input, 'mixnet-s', training=is_training_seg,
									   override_params=None, scope='segmentation')

		elif self.backbone == 'efficientnet':
			feature_list = EfficientNetB0(image_input, model_name='segmentation',
								 input_shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], num_ch),
								 keep_dropout_backbone=self.keep_dropout_backbone,
								 training=is_training_seg, include_top=True, classes=CLASS_NUM)

		elif self.backbone == 'fast_scnn':
			feature_list = build_fast_scnn(image_input, 'segmentation', is_training=is_training_seg,
										   keep_dropout_backbone=self.keep_dropout_backbone)

		elif self.backbone == 'ghostnet':
			# Set depth_multiplier to change the depth of GhostNet
			feature_list = ghostnet_base(image_input, scope='segmentation', dw_code=None, ratio_code=None,
									se=1, min_depth=8, depth=1, depth_multiplier=0.5, conv_defs=None,
									is_training=is_training_seg)
		elif self.backbone == 'sinet':
			feature_list = SINet(image_input, classes=CLASS_NUM, p=2, q=8, chnn=1, training=True,
												 bn_momentum=0.99, scope='segmentation', reuse=False)

		# create segmentation head

		if self.segmentation_head == 'bifpn':
			P3_out, P4_out, P5_out, P6_out, P7_out = bifpn_segmentation_head(feature_list, 64, scope='segmentation',
																			 is_training=is_training_seg)
			with tf.variable_scope('segmentation'):

				P3 = tf.layers.conv2d(P3_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P3')
				seg_out = tf.image.resize_images(P3, (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
				P4 = tf.layers.conv2d(P4_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P4')
				P5 = tf.layers.conv2d(P5_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P5')
				P6 = tf.layers.conv2d(P6_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P6')
				P7 = tf.layers.conv2d(P7_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P7')
				feature = [P3_out, P3]

			segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=seg_out, labels=mask)) + \
				tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P4, labels=tf.image.resize_images(mask, (IMAGE_SIZE[0]//4, IMAGE_SIZE[1]//4)))) + \
				tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P5, labels=tf.image.resize_images(mask, (IMAGE_SIZE[0]//8, IMAGE_SIZE[1]//8)))) + \
				tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P6, labels=tf.image.resize_images(mask, (IMAGE_SIZE[0]//16, IMAGE_SIZE[1]//16)))) + \
				tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P7, labels=tf.image.resize_images(mask, (IMAGE_SIZE[0]//32, IMAGE_SIZE[1]//32))))



		elif self.segmentation_head == 'fpn':

			feature = fpn_segmentation_head(feature_list, CLASS_NUM, drop_rate=0.2,
																keep_dropout_head=self.keep_dropout_head,
																scope='segmentation', training=is_training_seg)
			seg_out = tf.image.resize_images(feature[1], (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True)

			segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=seg_out, labels=mask))

		elif self.segmentation_head == 'bfp':

			P3_out, P4_out, P5_out, P6_out, P7_out = bfp_segmentation_head(feature_list, 64, scope='segmentation',
																		   is_training=is_training_seg)
			with tf.variable_scope('segmentation'):
				P3 = tf.layers.conv2d(P3_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P3')
				seg_out = tf.image.resize_images(P3, (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True)
				P4 = tf.layers.conv2d(P4_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P4')
				P5 = tf.layers.conv2d(P5_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P5')
				P6 = tf.layers.conv2d(P6_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P6')
				P7 = tf.layers.conv2d(P7_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P7')
			feature = [P3_out, P3]

			segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=seg_out, labels=mask)) + \
				tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P4, labels=tf.image.resize_images(mask, (IMAGE_SIZE[0]//4, IMAGE_SIZE[1]//4)))) + \
				tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P5, labels=tf.image.resize_images(mask, (IMAGE_SIZE[0]//8, IMAGE_SIZE[1]//8)))) + \
				tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P6, labels=tf.image.resize_images(mask, (IMAGE_SIZE[0]//16, IMAGE_SIZE[1]//16)))) + \
				tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=P7, labels=tf.image.resize_images(mask, (IMAGE_SIZE[0]//32, IMAGE_SIZE[1]//32))))
		else:
			feature = feature_list
			seg_out = tf.image.resize_images(feature[2], (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True)
			segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=seg_out, labels=mask))



		mask_out = tf.nn.sigmoid(feature[1], name='mask_out')

		# Create decision head
		dec_out = decision_head(feature[0], feature[1], class_num=1, scope='decision',
								keep_dropout_head=self.keep_dropout_head,
								training=is_training_dec)
		decision_out = tf.nn.sigmoid(dec_out, name='decision_out')

		decision_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_out, labels=label)
		decision_loss = tf.reduce_mean(decision_loss)
		total_loss = segmentation_loss + decision_loss

		# Variable list
		train_segment_var_list = [v for v in tf.trainable_variables() if 'segmentation' in v.name]

		train_decision_var_list = [v for v in tf.trainable_variables() if 'decision' in v.name]
		train_var_list = train_segment_var_list + train_decision_var_list

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		update_ops_segment = [v for v in update_ops if 'segmentation' in v.name]

		update_ops_decision = [v for v in update_ops if 'decision' in v.name]
		# optimizer_segmentation = tf.train.GradientDescentOptimizer(self.__learn_rate)
		optimizer_segmentation = tf.train.AdamOptimizer(self.__learn_rate)
		optimizer_decision = tf.train.AdamOptimizer(self.__learn_rate)
		# optimizer_decision = tf.train.GradientDescentOptimizer(self.__learn_rate)
		optimizer_total = tf.train.AdamOptimizer(self.__learn_rate)

		with tf.control_dependencies(update_ops_segment):
			optimize_segment = optimizer_segmentation.minimize(segmentation_loss, var_list=train_segment_var_list)
		with tf.control_dependencies(update_ops_decision):
			optimize_decision = optimizer_decision.minimize(decision_loss, var_list=train_decision_var_list)

		with tf.control_dependencies(update_ops):
			optimize_total = optimizer_total.minimize(total_loss, var_list=train_var_list)

		if not os.path.exists(self.tensorboard_logdir):
			os.makedirs(self.tensorboard_logdir)
		# merged = tf.summary.merge_all()
		# train_writer = tf.summary.FileWriter(self.tensorboard_logdir, self.session.graph)
		init_op = tf.global_variables_initializer()

		# self.update_ops_segment = update_ops_segment
		self.train_segment_var_list = [v for v in tf.trainable_variables() if 'segmentation' in v.name]
		self.feature_list = feature_list
		self.Image = image_input
		self.is_training_seg = is_training_seg
		self.is_training_dec = is_training_dec
		self.mask = mask
		self.label = label
		self.segmentation_loss = segmentation_loss
		self.decision_loss = decision_loss
		self.optimize_segment = optimize_segment
		self.optimize_decision = optimize_decision
		self.optimize_total = optimize_total
		self.decison_out = decision_out
		self.mask_out = mask_out
		self.init_op = init_op
		# self.train_writer = train_writer
		# self.merged = merged

	def save(self):
		self.__saver.save(
			self.session,
			os.path.join(self.__checkPoint_dir, 'ckp'),
			global_step=self.step
		)

	def save_PbModel(self):
		input_name = "Image"
		output_name = "decision_out"
		output_node_names = [input_name, output_name]
		print("模型保存为pb格式")
		output_graph_def = tf.graph_util.convert_variables_to_constants(self.session,
																		self.session.graph_def,
																		output_node_names)
		output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

		if not os.path.exists(self.__pb_model_path):
			os.makedirs(self.__pb_model_path)
		pbpath = os.path.join(self.__pb_model_path, 'frozen_inference_graph.pb')
		with tf.gfile.GFile(pbpath, mode='wb') as f:
			f.write(output_graph_def.SerializeToString())



	def freeze_session(self, keep_var_names=None, output_names=["decision_out"], clear_devices=True):
		from tensorflow.python.framework.graph_util import convert_variables_to_constants
		session = self.session
		graph = session.graph
		with graph.as_default():
			freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
			output_names = output_names or []
			output_names += [v.op.name for v in tf.global_variables()]
			input_graph_def = graph.as_graph_def()
			if clear_devices:
				for node in input_graph_def.node:
					node.device = ""
			frozen_graph = convert_variables_to_constants(session, input_graph_def,
														  output_names, freeze_var_names)
			return frozen_graph
