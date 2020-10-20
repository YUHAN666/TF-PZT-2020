""" implementation of LEDNet """
import tensorflow as tf
import numpy as np
from config import IMAGE_SIZE, CLASS_NUM


def channel_shuffle(input_tensor, groups):

    batch, h, w, inChannel = input_tensor.shape
    channelGroup = inChannel//groups

    x = tf.reshape(input_tensor, [batch, h, w, groups, channelGroup])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [batch, h, w, -1])

    return x


def downSampleBlock(input_tensor, outChannel, training, name):

    x1 = tf.layers.max_pooling2d(input_tensor, (2, 2), (2, 2), name=name+'_MaxPool')
    x2 = tf.layers.conv2d(input_tensor, outChannel, (3, 3), (2, 2), padding='same', use_bias=True, name=name+'_Conv2D')

    x = tf.concat([x1, x2], axis=-1, name=name+'_Concat')

    x = tf.layers.batch_normalization(x, training=training, name=name+'_BN')
    x = tf.nn.relu(x, name=name+'_ReLU')

    return x


def SS_nbt_module(input_tensor, outChannel, dilated, training, keep_prob=1.0, name=None, keep_dropout=True):

    # halfInChannel = input_tensor.shape[-1]//2
    # x1 = input_tensor[:, :, :, :halfInChannel]
    # x2 = input_tensor[:, :, :, halfInChannel:]
    x1, x2 = tf.split(input_tensor, 2, 3)

    channel1 = outChannel//2
    channel2 = outChannel - channel1
    x1 = tf.layers.conv2d(x1, filters=channel1, kernel_size=(3, 1), strides=(1, 1), padding='same', use_bias=True, name=name+'_x1Conv1')
    x1 = tf.nn.relu(x1)
    x1 = tf.layers.conv2d(x1, filters=channel1, kernel_size=(1, 3), strides=1, padding='same', use_bias=True, name=name+'_x1Conv2')
    x1 = tf.layers.batch_normalization(x1, training=training, name=name+'_x1BN1')
    x1 = tf.nn.relu(x1)
    x1 = tf.layers.conv2d(x1, filters=channel1, kernel_size=(3, 1), strides=1, padding='same', use_bias=True,
                          dilation_rate=(dilated, 1), name=name+'_x1Conv3')
    x1 = tf.nn.relu(x1)
    x1 = tf.layers.conv2d(x1, filters=channel1, kernel_size=(1, 3), strides=1, padding='same', use_bias=True,
                          dilation_rate=(1, dilated), name=name+'_x1Conv4')
    x1 = tf.layers.batch_normalization(x1, training=training, name=name+'_x1BN2')

    x2 = tf.layers.conv2d(x2, filters=channel2, kernel_size=(1, 3), strides=1, padding='same', use_bias=True, name=name+'_x2Conv1')
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.conv2d(x2, filters=channel2, kernel_size=(3, 1), strides=1, padding='same', use_bias=True, name=name+'_x2Conv2')
    x2 = tf.layers.batch_normalization(x2, training=training, name=name+'_x2BN1')
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.conv2d(x2, filters=channel2, kernel_size=(1, 3), strides=1, padding='same', use_bias=True,
                          dilation_rate=(1, dilated), name=name+'_x2Conv3')
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.conv2d(x2, filters=channel2, kernel_size=(3, 1), strides=1, padding='same', use_bias=True,
                          dilation_rate=(dilated, 1), name=name+'_x2Conv4')
    x2 = tf.layers.batch_normalization(x2, training=training, name=name+'_x2BN2')

    if keep_prob != 1.0 and keep_dropout:
        x1 = tf.nn.dropout(x1, keep_prob=keep_prob, name=name+'_x1dropout')
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob, name=name+'_x2dropout')

    x = tf.concat([x1, x2], axis=-1, name=name+'_concat')
    x = tf.nn.relu(input_tensor+x, name=name+'_ReLU')

    # return channel_shuffle(x, 2)
    return x


def encoder(input_tensor, training, keep_dropout):

    x = downSampleBlock(input_tensor, 31, training, 'downSample1')
    x = SS_nbt_module(x, 32, 1, training, 0.97, 'ssModule1', keep_dropout)
    x = SS_nbt_module(x, 32, 1, training, 0.97, 'ssModule2', keep_dropout)
    x = SS_nbt_module(x, 32, 1, training, 0.97, 'ssModule3', keep_dropout)

    x = downSampleBlock(x, 32, training, 'downSample2')
    x = SS_nbt_module(x, 64, 1, training, 0.97, 'ssModule4', keep_dropout)
    x = SS_nbt_module(x, 64, 1, training, 0.97, 'ssModule5', keep_dropout)

    x = downSampleBlock(x, 64, training, 'downSample3')
    x = SS_nbt_module(x, 128, 1, training, 0.7, 'ssModule6', keep_dropout)
    x = SS_nbt_module(x, 128, 2, training, 0.7, 'ssModule7', keep_dropout)
    x = SS_nbt_module(x, 128, 5, training, 0.7, 'ssModule8', keep_dropout)
    x = SS_nbt_module(x, 128, 9, training, 0.7, 'ssModule9', keep_dropout)

    x = SS_nbt_module(x, 128, 2, training, 0.7, 'ssModule10', keep_dropout)
    x = SS_nbt_module(x, 128, 5, training, 0.7, 'ssModule11', keep_dropout)
    x = SS_nbt_module(x, 128, 9, training, 0.7, 'ssModule12', keep_dropout)
    # x = SS_nbt_module(x, 128, 17, training, 0.7, 'ssModule13', keep_dropout)

    return x


def convBlock(input_tensor, outChannel, kernel_size, strides, padding, training, dilation_rate=(1, 1), use_bias=True):

    x = tf.layers.conv2d(input_tensor, outChannel, kernel_size, strides, padding,
                         dilation_rate=dilation_rate, use_bias=use_bias)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    return x


def apnModule(input_tensor, outChannel, training):

    h = input_tensor.shape[1]
    w = input_tensor.shape[2]

    b1 = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    b1 = tf.expand_dims(b1, axis=1)
    b1 = tf.expand_dims(b1, axis=1)
    b1 = convBlock(b1, outChannel, (1, 1), (1, 1), padding='valid', training=training)
    b1 = tf.image.resize_images(b1, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

    mid = convBlock(input_tensor, outChannel, (1, 1), (1, 1), padding='valid', training=training)

    x1 = convBlock(input_tensor, 1, (7, 7), (2, 2), padding='same', training=training)
    x2 = convBlock(x1, 1, (5, 5), (2, 2), padding='same', training=training)
    x3 = convBlock(x2, 1, (3, 3), (2, 2), padding='same', training=training)
    x3 = convBlock(x3, 1, (3, 3), (1, 1), padding='same', training=training)

    x3 = tf.image.resize_images(x3, (h//4, w//4), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    x2 = convBlock(x2, 1, (5, 5), (1, 1), padding='same', training=training)
    x = x2 + x3
    x = tf.image.resize_images(x, (h//2, w//2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    x1 = convBlock(x1, 1, (7, 7), (1, 1), padding='same', training=training)
    x = x + x1
    x = tf.image.resize_images(x, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

    x = tf.multiply(x, mid)
    x = x + b1

    return x


def decoder(input_tensor, training):

    x = apnModule(input_tensor, CLASS_NUM, training)

    return input_tensor, x


def lednet(input_tensor, training, scope, keep_dropout):

    with tf.variable_scope(scope):

        seg_fea = encoder(input_tensor, training, keep_dropout)

        # seg_out = convBlock(seg_fea, CLASS_NUM, (1, 1), (1, 1), padding='valid', training=training)
        #
        # mask = tf.image.resize_images(seg_out, (IMAGE_SIZE[0], IMAGE_SIZE[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        #                        align_corners=True)
        seg_fea, seg_out = decoder(seg_fea, training)

    return seg_fea, seg_out




if __name__ == "__main__":

    Image = tf.placeholder(tf.float32, (1, 928, 320, 32), name="image")

    # x = SS_nbt_module(Image, 128, 2, True, 0.3)
    # x = downSampleBlock(Image, 128, True, 'down')
    feature, seg_out = lednet(Image, training=True, scope='segmentation',
                              keep_dropout=True)


