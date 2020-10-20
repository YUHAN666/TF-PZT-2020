# coding=utf-8
import tensorflow as tf
import numpy as np
import cv2
import time
import os
from tensorflow.python.framework import graph_util
from data_manager import DataManager
from tensorflow.python.client import timeline
from config import IMAGE_SIZE, IMAGE_MODE
from tensorflow.python.platform import gfile

batch_size = 1


def listData(data_dir):
    """# list the files  of  the currtent  floder of  'data_dir'     ,subfoders are not included.
    :param data_dir:
    :return:  list of files
    """
    example_dirs = [x[1] for x in os.walk(data_dir)][0]
    example_lists = {os.path.basename(x[0]): x[2] for x in os.walk(data_dir)}
    data_list = []
    for i in range(len(example_dirs)):
        example_dir = example_dirs[i]
        example_list = example_lists[example_dir]
        # 过滤label图片
        example_list = [item for item in example_list if "label" not in item]
        for j in range(len(example_list)):
            example_image = data_dir + '/' + example_dir + '/' + example_list[j]
            data_list.append(example_image)

    return data_list


data_dir = "E:/CODES/FAST-SCNN/DATA/1pzt/"
data_list = listData(data_dir)
data_size = len(data_list)
iternum = data_size // batch_size
pb_path = "./pbMode/frozen_inference_graph_unfuse.pb"
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = False
# tf_config.allow_soft_placement = True

with gfile.FastGFile(pb_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())  # 定义输入的张量名称,对应网络结构的输入张量
        # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
        input_image_tensor = sess.graph.get_tensor_by_name("Image:0")
        # input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
        # input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
        # 定义输出的张量名称
        # output_mask = sess.graph.get_tensor_by_name("segment/sigmoid:0")
        output_class = sess.graph.get_tensor_by_name("decision_out:0")
        # 读取测试图片
        # im=read_image(image_path,resize_height,resize_width,normalization=True)
        # im = cv2.imread(image_path, 0)
        # im = cv2.resize(im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        # im = np.array(im[np.newaxis,:, :, np.newaxis])
        # im=im[np.newaxis,:]            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
        # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
        for i in range(iternum):
            for j in range(batch_size):
                img_path = data_list[i * batch_size + j]
                img = cv2.imread(img_path, IMAGE_MODE)
                img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
                if IMAGE_MODE == 0:
                    img = np.array(img[np.newaxis, :, :, np.newaxis])
                else:
                    img = np.array(img[np.newaxis, :, :, :])
                if (j == 0):
                    img_batch = img
                else:
                    img_batch = np.concatenate((img_batch, img), 0)

            start = time.process_time()
            whichclass = sess.run([output_class], feed_dict={input_image_tensor: img_batch},
                                        options=options, run_metadata=run_metadata)
            elapsed = (time.process_time() - start)
            print("time used:", elapsed)
            # mask = np.squeeze(mask) * 255
            print("out:{}".format(whichclass))

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('unfuse_timeline/timeline_02_step_%d_%d.json' % (i, j), 'w') as f:
                f.write(chrome_trace)