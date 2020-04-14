# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:01:58 2019

@author: RL
"""
import tensorflow as tf
import numpy as np
import cv2
import time
import os
from tensorflow.python.framework import graph_util
from data_manager import DataManager

IMAGE_SIZE=[144,144]

batch_size = 1 
IMAGE_MODE = 1



def freeze_graph_test(pb_path, img_batch):    
    '''    
    :param pb_path:pb文件的路径    
    :param image_path:测试图片的路径    
    :return:    '''    
    with tf.Graph().as_default():        
        output_graph_def = tf.GraphDef()        
        with open(pb_path, "rb") as f:            
            output_graph_def.ParseFromString(f.read())            
            tf.import_graph_def(output_graph_def, name="")        
            with tf.Session() as sess:            
                sess.run(tf.global_variables_initializer())             # 定义输入的张量名称,对应网络结构的输入张量            
                # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数            
                input_image_tensor = sess.graph.get_tensor_by_name("Image:0")            
                input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")            
                #input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")             
                # 定义输出的张量名称            
                output_mask = sess.graph.get_tensor_by_name("segment/sigmoid:0")
                output_class = sess.graph.get_tensor_by_name("decision/argmax:0")       
                # 读取测试图片            
                #im=read_image(image_path,resize_height,resize_width,normalization=True)
                #im = cv2.imread(image_path, 0)
                #im = cv2.resize(im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
                #im = np.array(im[np.newaxis,:, :, np.newaxis])
                #im=im[np.newaxis,:]            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字 
                # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})            
                start = time.process_time()
                whichclass, mask=sess.run([output_class,output_mask], feed_dict={input_image_tensor: img_batch,
                                                            input_keep_prob_tensor:1.0})  
                elapsed = (time.process_time()-start)          
                print("time used:",elapsed)
                mask = np.squeeze(mask)*255
                print("out:{}".format(whichclass))            
                return whichclass,mask

def listData(data_dir):
    """# list the files  of  the currtent  floder of  'data_dir'     ,subfoders are not included.
    :param data_dir:
    :return:  list of files
    """
    example_dirs = [x[1] for x in os.walk(data_dir)][0]
    example_lists = {os.path.basename(x[0]): x[2] for x in os.walk(data_dir)}
    data_list=[]
    for i in range(len(example_dirs)):
        example_dir = example_dirs[i]
        example_list = example_lists[example_dir]
        # 过滤label图片
        example_list = [item for item in example_list if "label" not in item]
        for j in range(len(example_list)):
            example_image = data_dir+'/'+example_dir + '/' + example_list[j]
            data_list.append(example_image)               

    return data_list
        
 


data_dir = "D:/chen_lin/test/"
data_list = listData(data_dir) 
data_size = len(data_list)
iternum = data_size//batch_size
pb_path = "D:/DL/OCR2/pb/OCR0917.pb"
#tf_config = tf.ConfigProto()
#tf_config.gpu_options.allow_growth = False
#tf_config.allow_soft_placement = True



with tf.Graph().as_default():        
    output_graph_def = tf.GraphDef()        
    with open(pb_path, "rb") as f:            
        output_graph_def.ParseFromString(f.read())            
        tf.import_graph_def(output_graph_def, name="")        
        with tf.Session() as sess:            
            sess.run(tf.global_variables_initializer())             # 定义输入的张量名称,对应网络结构的输入张量            
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数            
            input_image_tensor = sess.graph.get_tensor_by_name("input_1:0")            
                     
            #input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")             
            # 定义输出的张量名称            
            
            output_class = sess.graph.get_tensor_by_name("dense_1/Softmax:0")       
            # 读取测试图片            
            #im=read_image(image_path,resize_height,resize_width,normalization=True)
            #im = cv2.imread(image_path, 0)
            #im = cv2.resize(im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            #im = np.array(im[np.newaxis,:, :, np.newaxis])
            #im=im[np.newaxis,:]            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字 
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False}) 
            for i in range(iternum):
                for j in range(batch_size):
                    img_path = data_list[i*batch_size+j]
                    img = cv2.imread(img_path, IMAGE_MODE)
                    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
                    img = (img/255-0.5)*2
                    if IMAGE_MODE==0:
                        img = np.array(img[np.newaxis,:, :, np.newaxis])
                    else:
                        img = np.array(img[np.newaxis,:, :,:])
                    if(j==0):
                        img_batch = img
                    else:
                        img_batch = np.concatenate((img_batch,img),0)
               
    
                start = time.process_time()
                whichclass =sess.run(output_class, feed_dict={input_image_tensor: img_batch})  
                whichclass = np.argmax(whichclass)
                elapsed = (time.process_time()-start)          
                print("time used:",elapsed)
               
                print("out:{}".format(whichclass))            
             
    
#input_checkpoint='D:/DL/SDD_fast_scnn 0830_2/checkpoint/ckp-620'    # 输出pb模型的路径    
#out_pb_path="D:/DL/SDD_fast_scnn 0830_2/pbMode/frozen_inference_graph.pb"    # 调用freeze_graph将ckpt转为pb    
#
#
#image_path = 'D:/DL/SDD_fast_scnn0827/Datasets/KolektorSDD/kos38/Part0.jpg'
#
#wichclass,mask = freeze_graph_test(pb_path=out_pb_path, image_path=image_path)