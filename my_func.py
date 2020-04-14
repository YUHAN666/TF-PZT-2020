# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:56:40 2019

@author: RL
"""
CLASS_NUM = 13
import tensorflow as tf
import cv2
from tensorflow.python.ops import math_ops


def conv_block(inputs, conv_type, filters, kernel_size, strides, training,name=None,padding='same',relu=True,use_bias=False):
    if(conv_type == 'ds'):            
        x = tf.layers.separable_conv2d(inputs,filters,kernel_size,strides,name = name + '_ds_conv',padding=padding,use_bias=use_bias)
    else:             
        x = tf.layers.conv2d(inputs,filters,kernel_size,strides,name=name+'_conv2d',padding = 'same',use_bias=use_bias)       
    x = tf.layers.batch_normalization(x,training=training,name = name+'_bn')   
    
    if (relu):
        x = tf.nn.relu(x,name = name+'_relu')
    return x

def _res_bottleneck(inputs, filters, kernel, t, s,training,name = None,res=False):
    
   
    tchannel = inputs.get_shape().as_list()[-1]*t  
    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1),name = name+'_conv',training=training)     
     
    x = tf.keras.layers.DepthwiseConv2D(kernel, s, depth_multiplier=1, padding='same')(x)    
    x = tf.layers.batch_normalization(x,training=training,name = name+'_bn')
    #x = bn_layer(x,is_training = training,scope = name+'_bn')
    x = tf.nn.relu(x,name = name+'relu')      
    x = conv_block(x, 'conv', filters, (1, 1), strides=1,training=training,name = name+'_conv2',padding='same',relu=False,use_bias = False)    
    if res:        
        x = x+ inputs
    return x

def bottleneck_block(inputs, filters, kernel, t, strides, n,training,name = None):
    x = _res_bottleneck(inputs, filters, kernel, t, strides,training,name = name+'_res')
    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, training,name = name+str(i)+'_res',res=True)
    return x

def Classifier(inputs,training,scope = 'classifier',reuse = None):
    with tf.variable_scope(scope, reuse=reuse):  
        x = conv_block(inputs, 'ds', 32, (3, 3), strides = (1, 1),training=training,name =  'dsconv1',padding='same')
        x = tf.layers.max_pooling2d(x,(2,2),(2,2),padding = 'same')
        x = conv_block(x, 'ds', 64, (3, 3), strides = (1, 1),training=training,name = 'dsconv2',padding='same')    
        x = tf.layers.max_pooling2d(x,(2,2),(2,2),padding = 'same')        
        x = conv_block(x, 'conv', 64, (1, 1), strides=(1, 1),training=training, name = 'conv3',padding='same', relu=True)
        vector=math_ops.reduce_mean(x,[1,2],name='gap', keepdims=True)
        vector=tf.squeeze(vector,axis=[1,2])
        logits = tf.layers.dense(vector,CLASS_NUM)
        final_tensor = tf.nn.softmax(logits)
        return logits,final_tensor

def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4

    M = cv2.getRotationMatrix2D(center, angle, scale) #5

    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7
