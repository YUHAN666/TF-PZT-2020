import os
import numpy as np
import tensorflow as tf
from config import CLASS_NUM, IMAGE_SIZE, TRAIN_MODE_IN_TRAIN, TRAIN_MODE_IN_TEST, TRAIN_MODE_IN_VALID, IMAGE_MODE, DATA_FORMAT, ACTIVATION
from decision_head import decision_head
from tensorflow.python import pywrap_tensorflow


class Model(object):

    def __init__(self, sess, param):
        self.step = 0
        self.session = sess
        self.__learn_rate = param["learn_rate"]
        self.__max_to_keep = param["max_to_keep"]
        self.__pb_model_path = param["pb_Mode_dir"]
        self.__restore = param["b_restore"]
        self.__bn_momentum = param["momentum"]
        self.__mode = param["mode"]
        # self.__mode = "testing"
        self.backbone = param["backbone"]
        self.tensorboard_logdir = param["tensorboard_logdir"]
        self.neck = param["neck"]
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
            self.is_pb = True
        else:
            self.is_pb = False

        # Building graph
        with self.session.as_default():
            self.build_model()
        # 参数初始化，或者读入参数
        with self.session.as_default():
            self.init_op.run()
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.__saver = tf.train.Saver(var_list, max_to_keep=self.__max_to_keep)
            # Loading last save if needed
            if self.__restore:
                ckpt = tf.train.latest_checkpoint(self.__checkPoint_dir)
                if ckpt:

                    var_list2 = [v for v in var_list if "bias" not in v.name]

                    self.__saver2 = tf.train.Saver(var_list2, max_to_keep=self.__max_to_keep)

                    self.step = int(ckpt.split('-')[1])
                    self.__saver2.restore(self.session, ckpt)
                    print('Restoring from epoch:{}'.format(self.step))
                    self.step += 1

                    if self.__mode == 'savePb':
                        reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
                        var_to_shape_map = reader.get_variable_to_shape_map()
                        source_list = [key for key in var_to_shape_map if "CBR" in key]
                        epsilon = 0.001
                        for key in source_list:
                            if "moving_mean" in key:
                                mean = np.array(reader.get_tensor(key))

                                key_var = key[0:-11] + "moving_variance"
                                var = np.array(reader.get_tensor(key_var))

                                key_gamma = key[0:-11] + "gamma"
                                gamma = np.array(reader.get_tensor(key_gamma))

                                key_beta = key[0:-11] + "beta"
                                beta = np.array(reader.get_tensor(key_beta))

                                key_W = key[0:-14] + "Conv2D/kernel"
                                W = np.array(reader.get_tensor(key_W))

                                alpha = gamma / ((var + epsilon) ** 0.5)

                                W_new = W * alpha

                                B_new = beta - mean * alpha

                                weight = tf.get_default_graph().get_tensor_by_name(key_W + ':0')

                                update_weight = tf.assign(weight, W_new)

                                bias_name = key_W[0:-6] + 'bias:0'

                                bias = tf.get_default_graph().get_tensor_by_name(bias_name)

                                updata_bias = tf.assign(bias, B_new)

                                sess.run(update_weight)
                                sess.run(updata_bias)

    def build_model(self):

        # tf.summary.image('input_image', image_input, 10)
        # tf.summary.image('mask', mask, 10)

        if (self.is_pb == False):
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
            from MixNet import MixNetSmall
            self.__checkPoint_dir = 'checkpoint/mixnet'
            backbone_output = MixNetSmall(image_input, scope='segmentation', include_top=False,
                                          keep_dropout_backbone=self.keep_dropout_backbone, training=is_training_seg)

        elif self.backbone == 'mixnet_official':
            from MixNet_official import build_model_base
            self.__checkPoint_dir = 'checkpoint/mixnet_official'
            backbone_output = build_model_base(image_input, 'mixnet-s', training=is_training_seg,
                                               override_params=None, scope='segmentation')

        elif self.backbone == 'efficientnet':
            from efficientnet import EfficientNetB0
            self.__checkPoint_dir = 'checkpoint/efficientnet'
            backbone_output = EfficientNetB0(image_input, model_name='segmentation',
                                             input_shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], num_ch),
                                             keep_dropout_backbone=self.keep_dropout_backbone,
                                             training=is_training_seg, include_top=True, classes=CLASS_NUM)

        elif self.backbone == 'fast_scnn':
            from fast_scnn import build_fast_scnn
            self.__checkPoint_dir = 'checkpoint/fast_scnn'
            backbone_output = build_fast_scnn(image_input, 'segmentation', is_training=is_training_seg,
                                              keep_dropout_backbone=self.keep_dropout_backbone)
        elif self.backbone == 'ghostnet':
            from GhostNet import ghostnet_base
            self.__checkPoint_dir = 'checkpoint/ghostnet'
            # Set depth_multiplier to change the depth of GhostNet
            backbone_output = ghostnet_base(image_input, mode=self.__mode, data_format=DATA_FORMAT, scope='segmentation',
                                            dw_code=None, ratio_code=None,
                                            se=1, min_depth=8, depth=1, depth_multiplier=0.5, conv_defs=None,
                                            is_training=is_training_seg, momentum=self.__bn_momentum)

        elif self.backbone == 'sinet':
            from SiNet import SINet
            self.__checkPoint_dir = 'checkpoint/sinet'
            backbone_output = SINet(image_input, classes=CLASS_NUM, p=2, q=8, chnn=1, training=True,
                                    bn_momentum=0.99, scope='segmentation', reuse=False)

        elif self.backbone == 'lednet':
            from LEDNet import lednet
            self.__checkPoint_dir = 'checkpoint/lednet'
            backbone_output = lednet(image_input, training=is_training_seg, scope='segmentation',
                                     keep_dropout=False)

        elif self.backbone == 'cspnet':
            from CSPDenseNet import CSPPeleeNet
            self.__checkPoint_dir = 'checkpoint/cspnet'
            backbone_output = CSPPeleeNet(image_input, data_format=DATA_FORMAT, drop_rate=0.0,
                                          training=self.is_training, momentum=self.__bn_momentum,
                                          name="segmentation", mode=self.__mode, activation=ACTIVATION)
        elif self.backbone == 'FCNN':
            self.__checkPoint_dir = 'checkpoint/fcnn'
            from FCNN import conv_block, LearningToDownsample,GlobalFeatureExtractor,FeatureFusion,Classifier

            def SegmentNet(input, scope, is_training, reuse=None):
                with tf.variable_scope(scope, reuse=None):
                    lds2, lds1 = LearningToDownsample(input, is_training, scope='lds', reuse=reuse)
                    gfe = GlobalFeatureExtractor(lds2, is_training, scope='gfe', reuse=reuse)
                    ff = FeatureFusion(lds1, lds2, gfe, is_training, scope='ff', reuse=reuse)
                    features, logits = Classifier(ff, is_training, scope='classify', reuse=reuse)
                    mask = tf.nn.sigmoid(logits, name='softmax1')
                    return features,logits,mask

            features, logits_pixel, mask = SegmentNet(image_input, 'segmentation', self.is_training)

            if self.is_pb == True:
                self.mask_out_l = tf.nn.sigmoid(logits_pixel[0], name='mask_out1')
                self.mask_out_r = tf.nn.sigmoid(logits_pixel[1], name='mask_out2')


            if self.is_pb == False:
                # Variable list
                segmentation_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=mask))
                train_segment_var_list = [v for v in tf.trainable_variables() if 'segmentation' in v.name]

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                update_ops_segment = [v for v in update_ops if 'segmentation' in v.name]

                optimizer_segmentation = tf.train.AdamOptimizer(self.__learn_rate)

                with tf.control_dependencies(update_ops_segment):
                    optimize_segment = optimizer_segmentation.minimize(segmentation_loss,
                                                                       var_list=train_segment_var_list)

                self.segmentation_loss = segmentation_loss
                self.optimize_segment = optimize_segment
            mask_out = tf.nn.sigmoid(logits_pixel, name='mask_out')
            self.mask_out = mask_out
            init_op = tf.global_variables_initializer()
            self.Image = image_input
            self.is_training_seg = is_training_seg
            self.is_training_dec = is_training_dec
            self.mask = mask
            self.label = label

            self.init_op = init_op

            return

        else:
            raise ValueError("Unknown Backbone")

        # create segmentation head and segmentation loss

        if len(backbone_output) == 5:
            if self.neck == 'bifpn':
                from FPN import bifpn_neck
                P3_out, P4_out, P5_out, P6_out, P7_out = bifpn_neck(backbone_output, 64, scope='segmentation',
                                                                    is_training=is_training_seg, momentum=self.__bn_momentum, mode=self.__mode, data_format=DATA_FORMAT)
                with tf.variable_scope('segmentation'):

                    P3 = tf.layers.conv2d(P3_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P3', data_format=DATA_FORMAT)
                    P4 = tf.layers.conv2d(P4_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P4', data_format=DATA_FORMAT)
                    P5 = tf.layers.conv2d(P5_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P5', data_format=DATA_FORMAT)
                    P6 = tf.layers.conv2d(P6_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P6', data_format=DATA_FORMAT)
                    P7 = tf.layers.conv2d(P7_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P7', data_format=DATA_FORMAT)
                    neck_output = [P3, P4, P5, P6, P7]
                    decision_in = [P3_out, P3]
            elif self.neck == 'fpn':
                from FPN import fpn_neck

                neck_output = fpn_neck(backbone_output, CLASS_NUM, drop_rate=0.2, keep_dropout_head=self.keep_dropout_head,
                                       scope='segmentation', training=is_training_seg)
            elif self.neck == 'bfp':
                from BFP import bfp_segmentation_head

                P3_out, P4_out, P5_out, P6_out, P7_out = bfp_segmentation_head(backbone_output, 64, scope='segmentation',
                                                                               is_training=is_training_seg)
                with tf.variable_scope('segmentation'):
                    P3 = tf.layers.conv2d(P3_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P3')
                    P4 = tf.layers.conv2d(P4_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P4')
                    P5 = tf.layers.conv2d(P5_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P5')
                    P6 = tf.layers.conv2d(P6_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P6')
                    P7 = tf.layers.conv2d(P7_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P7')
                neck_output = [P3, P4, P5, P6, P7]
                decision_in = [P3_out, P3]
            elif self.neck == 'pan':
                from FPN import PAN
                with tf.variable_scope('segmentation'):
                    backbone_output = PAN(backbone_output, 128, training=self.is_training_seg)

                    seg_fea = backbone_output[-1]
                    seg_fea = tf.keras.layers.UpSampling2D((4, 4))(seg_fea)
                    seg_fea = tf.keras.layers.DepthwiseConv2D((3, 3), (1, 1), padding='same')(seg_fea)
                    seg_fea = tf.layers.batch_normalization(seg_fea, training=is_training_seg)
                    seg_fea = tf.nn.relu(seg_fea)
                    seg_fea = tf.layers.conv2d(seg_fea, 128, (1, 1))

                    seg_fea = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same')(seg_fea)
                    seg_fea = tf.layers.batch_normalization(seg_fea, training=is_training_seg)
                    seg_fea = tf.nn.relu(seg_fea)

                    seg_fea = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same')(seg_fea)
                    seg_fea = tf.layers.batch_normalization(seg_fea, training=is_training_seg)
                    seg_fea = tf.nn.relu(seg_fea)

                    seg_fea1 = tf.layers.conv2d(seg_fea, 1, (1, 1))

                    neck_output = [seg_fea, seg_fea1]
            else:
                raise ValueError(" Unknown neck ")

            if len(neck_output) == 5:
                if DATA_FORMAT == 'channels_first':
                    for nec_index in range(len(neck_output)):
                        neck_output[nec_index] = tf.transpose(neck_output[nec_index], [0, 2, 3, 1])
                logits_pixel = tf.image.resize_images(neck_output[0], (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True,
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=mask)) + \
                                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neck_output[1], labels=tf.image.resize_images(mask, (IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4)))) + \
                                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neck_output[2], labels=tf.image.resize_images(mask, (IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8)))) + \
                                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neck_output[3], labels=tf.image.resize_images(mask, (IMAGE_SIZE[0] // 16, IMAGE_SIZE[1] // 16)))) + \
                                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neck_output[4], labels=tf.image.resize_images(mask, (IMAGE_SIZE[0] // 32, IMAGE_SIZE[1] // 32))))

            elif len(neck_output) == 2:
                decision_in = neck_output
                if DATA_FORMAT == 'channels_first':
                    for nec_index in range(len(neck_output)):
                        neck_output[nec_index] = tf.transpose(neck_output[nec_index], [0, 2, 3, 1])
                logits_pixel = tf.image.resize_images(neck_output[1], (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True)
                segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=mask))
        elif len(backbone_output) == 2:
            decision_in = [backbone_output[0], backbone_output[1]]
            if DATA_FORMAT == 'channels_first':
                for nec_index in range(len(backbone_output)):
                    backbone_output[nec_index] = tf.transpose(backbone_output[nec_index], [0, 2, 3, 1])
            logits_pixel = tf.image.resize_images(backbone_output[1], (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True)

            segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=mask))
        else:
            raise ValueError(" Incorrect backbone output number, must be 3 or 5")

        mask_out = tf.nn.sigmoid(logits_pixel, name='mask_out')
        if self.is_pb == True:
            self.mask_out_l = tf.nn.sigmoid(logits_pixel[0], name='mask_out1')
            self.mask_out_r = tf.nn.sigmoid(logits_pixel[1], name='mask_out2')

        # Create decision head
        dec_out = decision_head(decision_in[0], decision_in[1], class_num=CLASS_NUM, scope='decision',
                                keep_dropout_head=self.keep_dropout_head,
                                training=is_training_dec, data_format=DATA_FORMAT, momentum=self.__bn_momentum,
                                mode=self.__mode, activation=ACTIVATION)
        decision_out = tf.nn.sigmoid(dec_out, name='decision_out')
        decision_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_out, labels=label)
        decision_loss = tf.reduce_mean(decision_loss)
        total_loss = segmentation_loss + decision_loss

        if self.is_pb == False:
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

            self.segmentation_loss = segmentation_loss
            self.decision_loss = decision_loss
            self.optimize_segment = optimize_segment
            self.optimize_decision = optimize_decision
            self.optimize_total = optimize_total

        if not os.path.exists(self.tensorboard_logdir):
            os.makedirs(self.tensorboard_logdir)
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(self.tensorboard_logdir, self.session.graph)
        init_op = tf.global_variables_initializer()

        # self.update_ops_segment = update_ops_segment
        # self.train_segment_var_list = [v for v in tf.trainable_variables() if 'segmentation' in v.name]
        # self.backbone_output = backbone_output
        self.Image = image_input
        self.is_training_seg = is_training_seg
        self.is_training_dec = is_training_dec
        self.mask = mask
        self.label = label
        self.decison_out = decision_out
        self.mask_out = mask_out
        self.init_op = init_op
    # self.train_writer = train_writer
    # self.merged = merged

    def save(self):
        if not os.path.exists(self.__checkPoint_dir):
            os.makedirs(self.__checkPoint_dir)

        self.__saver.save(
            self.session,
            os.path.join(self.__checkPoint_dir, 'ckp'),
            global_step=self.step
        )


    def save_PbModel(self):
        input_name = "Image"
        output_name = "decision_out"
        output_node_names = [input_name, output_name]
        # output_node_names = [input_name]
        output_node_names.append("mask_out1")
        output_node_names.append("mask_out2")
        print("模型保存为pb格式")
        output_graph_def = tf.graph_util.convert_variables_to_constants(self.session,
                                                                        self.session.graph_def,
                                                                        output_node_names)
        output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

        if not os.path.exists(self.__pb_model_path):
            os.makedirs(self.__pb_model_path)
        pbpath = os.path.join(self.__pb_model_path, 'frozen_inference_graph_fuse_glue2.pb')
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
