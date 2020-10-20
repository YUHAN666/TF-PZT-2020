import tensorflow as tf
import numpy as np
from config import CLASS_NUM, BIN_SIZE
from convblock import ConvBatchNormRelu as CBR
from tensorflow.python.ops import math_ops
slim=tf.contrib.slim

pelee_cfg = {
    "num_init_features": 32,
    "growthRate": 16,
    "nDenseBlocks": [2, 4, 8, 6],
    "bottleneck_width": [1, 2, 4, 4]
}


def StemBlock(inputs, num_init_features, data_format, training, momentum, mode):
    channel_axis = 1 if data_format == 'channels_first' else -1

    x = CBR(inputs, num_init_features, 3, 2, training=training, momentum=momentum, name="CBR1", data_format=data_format,
            mode=mode)
    x1 = CBR(x, num_init_features // 2, 1, 1, training=training, momentum=momentum, name="1x1CBR_R",
             data_format=data_format, mode=mode)
    x1 = CBR(x1, num_init_features, 3, 2, training=training, momentum=momentum, name="3x3CBR_R",
             data_format=data_format, mode=mode)

    x2 = tf.layers.max_pooling2d(x, (2, 2), (2, 2), data_format=data_format, padding='same')
    out = tf.concat([x1, x2], axis=channel_axis)

    out = CBR(out, num_init_features, 1, 1, training=training, momentum=momentum, name="CBR2", data_format=data_format,
              mode=mode)

    return out


def DenseBlock(inputs, inter_channel, growth_rate, data_format, training, momentum, name, mode):
    with tf.variable_scope(name, reuse=False):
        channel_axis = 1 if data_format == 'channels_first' else -1

        x1 = CBR(inputs, inter_channel, 1, 1, training=training, momentum=momentum, name="1x1CBR_R",
                 data_format=data_format, mode=mode)
        x1 = CBR(x1, growth_rate, 3, 1, training=training, momentum=momentum, name="3x3CBR_R", data_format=data_format,
                 mode=mode)
        out = tf.concat([inputs, x1], axis=channel_axis)

        return out


def pyramid_pooling_block(input_tensor, nOut, bin_sizes, training, momentum, data_format='channels_first', name=None,
                          mode=None):
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

        x = CBR(x, outlist[n - 1], (1, 1), strides=(1, 1), padding='valid', name=name + '_' + str(n) + 'conv',
                training=training, momentum=momentum, data_format=data_format, mode=mode)

        if data_format == 'channels_last':
            x = tf.image.resize_images(x, (h, w), align_corners=True)
        else:
            x = tf.transpose(x, [0, 2, 3, 1])  # NCHW->NHWC
            x = tf.image.resize_images(x, (h, w), align_corners=True)
            x = tf.transpose(x, [0, 3, 1, 2])  # NHWC -> NCHW
        concat_list.append(x)

    x = tf.concat(concat_list, axis=axis)

    x = CBR(x, nOut, (1, 1), strides=(1, 1), training=training, momentum=momentum,
            name=name + 'conv', padding='valid', data_format=data_format, mode=mode)

    return x


def CSPPeleeNet(inputs, data_format, drop_rate, training, momentum, name, pelee_cfg=pelee_cfg, mode=None, activation='relu'):
    with tf.variable_scope(name, reuse=False):

        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])  # NHWC -> NCHW

        channel_axis = 1 if data_format == 'channels_first' else -1

        num_init_features = pelee_cfg["num_init_features"]
        growthRate = pelee_cfg["growthRate"]
        # half_growth_rate =  growthRate// 2
        nDenseBlocks = pelee_cfg["nDenseBlocks"]
        bottleneck_width = pelee_cfg["bottleneck_width"]

        x = StemBlock(inputs, num_init_features, data_format, training, momentum, mode=mode)

        inter_channel = list()
        total_filter = list()
        dense_inp = list()
        for i, b_w in enumerate(bottleneck_width):

            inter_channel.append(int(growthRate * b_w / 4) * 4)

            if i == 0:
                total_filter.append(num_init_features + growthRate * nDenseBlocks[i])
                dense_inp.append(num_init_features)
            else:
                total_filter.append(total_filter[i - 1] + growthRate * nDenseBlocks[i])
                dense_inp.append(total_filter[i - 1])
            relu = activation if i == 0 else None

            x1 = CBR(x, dense_inp[i], 1, 1, training=training, momentum=momentum, name="split_conv_L" + str(i),
                     data_format=data_format, activation=relu, mode=mode)
            x2 = CBR(x, dense_inp[i], 1, 1, training=training, momentum=momentum, name="split_conv_R" + str(i),
                     data_format=data_format, activation=relu, mode=mode)

            x = x1
            for n in range(nDenseBlocks[i]):
                x = DenseBlock(x, inter_channel[i], growthRate, data_format, training, momentum,
                               name="Denseblock" + str(i) + "_" + str(n), mode=mode)

                # transition layer-1
            x = CBR(x, total_filter[i], 1, 1, training=training, momentum=momentum, name="transition_1_" + str(i),
                    data_format=data_format, mode=mode)
            x = tf.concat([x, x2], axis=channel_axis)
            # transition layer-2
            x = CBR(x, total_filter[i], 1, 1, training=training, momentum=momentum, name="transition_2_" + str(i),
                    data_format=data_format, mode=mode)

            if i != len(nDenseBlocks) - 1:
                x = tf.layers.AveragePooling2D(pool_size=2, strides=2, name='agp' + str(i), data_format=data_format)(x)
                if i == 0:
                    hi_res = x

        x = pyramid_pooling_block(x, total_filter[-1], BIN_SIZE,
                                  training=training, momentum=momentum, data_format=data_format, name='ppb', mode=mode)

        x = CBR(x, total_filter[-1], 1, 1, training=training, momentum=momentum, name="low_res_conv",
                data_format=data_format, activation=None, mode=mode)

        x = tf.keras.layers.UpSampling2D((4, 4), data_format=data_format)(x)

        hi_res = CBR(hi_res, total_filter[0], 1, 1, training=training, momentum=momentum, name="hi_res_conv",
                     data_format=data_format, activation=None, mode=mode)

        x = tf.concat([x, hi_res], axis=channel_axis)

        # x = x+hi_res

        x = CBR(x, 128, 1, 1, training=training, momentum=momentum, name="mix_conv", data_format=data_format, activation=activation,
                mode=mode)

        features = tf.layers.dropout(x, drop_rate, training=training, name='dropout')

        logits = CBR(features, CLASS_NUM, 1, 1, training=training, momentum=momentum, name="classify_conv",
                     data_format=data_format, activation=None, mode=mode)

        return [features, logits]


