# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:22:26 2019

@author: m088023
"""

import tensorflow as tf

sess=tf.Session
with tf.Graph().as_default():
    with tf.gfile.GFile("D:/DL/OCR/pb/OCR2.pb", 'rb') as modelfile:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(modelfile.read())
        tf.import_graph_def(graph_def)
        [print(n.name) for n in tf.get_default_graph().as_graph_def().node]