# -*- coding: utf-8 -*-
"""
Implementation of MixNet
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from typing import List
import re


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """
    Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.
    This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, depth_multiplier, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(depth_multiplier) if depth_multiplier is not None else None
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def round_repeats(repeats):
    """Round number of filters based on depth multiplier."""
    return int(repeats)


class BlockArgs(object):

    def __init__(self, input_filters=None,
                 output_filters=None,
                 dw_kernel_size=None,
                 expand_kernel_size=None,
                 project_kernel_size=None,
                 strides=None,
                 num_repeat=None,
                 se_ratio=None,
                 expand_ratio=None,
                 identity_skip=True,
                 swish=False,
                 dilated=False):

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.dw_kernel_size = self._normalize_kernel_size(dw_kernel_size)
        self.expand_kernel_size = self._normalize_kernel_size(expand_kernel_size)
        self.project_kernel_size = self._normalize_kernel_size(project_kernel_size)
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.identity_skip = identity_skip
        self.swish = swish
        self.dilated = dilated

    def decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        self.input_filters = int(options['i'])
        self.output_filters = int(options['o'])
        self.dw_kernel_size = self._parse_ksize(options['k'])
        self.expand_kernel_size = self._parse_ksize(options['a'])
        self.project_kernel_size = self._parse_ksize(options['p'])
        self.num_repeat = int(options['r'])
        self.identity_skip = ('noskip' not in block_string)
        self.se_ratio = float(options['se']) if 'se' in options else None
        self.expand_ratio = int(options['e'])
        self.strides = [int(options['s'][0]), int(options['s'][1])]
        self.swish = 'sw' in block_string
        self.dilated = 'dilated' in block_string

        return self

    def encode_block_string(self, block):
        """Encodes a block to a string.

        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9
         - {} encapsulates optional arguments

        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```
        """

        args = [
            'r%d' % block.num_repeat,
            'k%s' % self._encode_ksize(block.kernel_size),
            'a%s' % self._encode_ksize(block.expand_kernel_size),
            'p%s' % self._encode_ksize(block.project_kernel_size),
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]

        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)

        if block.id_skip is False:
            args.append('noskip')

        if block.swish:
            args.append('sw')

        if block.dilated:
            args.append('dilated')

        return '_'.join(args)

    def _normalize_kernel_size(self, val):
        if type(val) == int:
            return [val]

        return val

    def _parse_ksize(self, ss):
        return [int(k) for k in ss.split('.')]

    def _encode_ksize(self, arr):
        return '.'.join([str(k) for k in arr])

    @classmethod
    def from_block_string(cls, block_string):
        """
        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9
         - {} encapsulates optional arguments

        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```

        Returns:
            BlockArgs object initialized with the block
            string args.
        """
        block = cls()
        return block.decode_block_string(block_string)


# Default list of blocks for MixNets
def get_mixnet_small(depth_multiplier=None):
    # blocks_args = [
    #     'r1_k3_a1_p1_s11_e1_i16_o16',
    #     'r1_k3_a1.1_p1.1_s22_e6_i16_o24',
    #     'r1_k3_a1.1_p1.1_s11_e3_i24_o24',
    #
    #     'r1_k3.5.7_a1_p1_s22_e6_i24_o40_se0.5_sw',
    #     'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',
    #
    #     'r1_k3.5.7_a1_p1.1_s22_e6_i40_o80_se0.25_sw',
    #     'r2_k3.5_a1_p1.1_s11_e6_i80_o80_se0.25_sw',
    #
    #     'r1_k3.5.7_a1.1_p1.1_s11_e6_i80_o120_se0.5_sw',
    #     'r2_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',
    #
    #     'r1_k3.5.7.9.11_a1_p1_s22_e6_i120_o200_se0.5_sw',
    #     'r2_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
    # ]

    blocks_args = [
        'r1_k3_a1_p1_s11_e1_i16_o16',
        'r1_k3_a1.1_p1.1_s22_e6_i16_o24',
        'r1_k3_a1.1_p1.1_s11_e3_i24_o24',

        'r1_k3.5.7_a1_p1_s22_e6_i24_o40_se0.5',
        'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5',

        'r1_k3.5.7_a1_p1.1_s22_e6_i40_o80_se0.25',
        'r2_k3.5_a1_p1.1_s11_e6_i80_o80_se0.25',

        'r1_k3.5.7_a1.1_p1.1_s11_e6_i80_o120_se0.5',
        'r2_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5',

        'r1_k3.5.7.9.11_a1_p1_s22_e6_i120_o200_se0.5',
        'r2_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5',
    ]

    DEFAULT_BLOCK_LIST = [BlockArgs.from_block_string(s)
                          for s in blocks_args]

    return DEFAULT_BLOCK_LIST


def get_mixnet_medium(depth_multiplier=None):
    blocks_args = [
        'r1_k3_a1_p1_s11_e1_i24_o24',
        'r1_k3.5.7_a1.1_p1.1_s22_e6_i24_o32',
        'r1_k3_a1.1_p1.1_s11_e3_i32_o32',

        'r1_k3.5.7.9_a1_p1_s22_e6_i32_o40_se0.5_sw',
        'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

        'r1_k3.5.7_a1_p1_s22_e6_i40_o80_se0.25_sw',
        'r3_k3.5.7.9_a1.1_p1.1_s11_e6_i80_o80_se0.25_sw',

        'r1_k3_a1_p1_s11_e6_i80_o120_se0.5_sw',
        'r3_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

        'r1_k3.5.7.9_a1_p1_s22_e6_i120_o200_se0.5_sw',
        'r3_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
    ]

    DEFAULT_BLOCK_LIST = [BlockArgs.from_block_string(s)
                          for s in blocks_args]

    return DEFAULT_BLOCK_LIST


def get_mixnet_large(depth_multiplier=None):
    return get_mixnet_medium(depth_multiplier)


def split_channels(total_filters, num_groups):
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


def swish_activation(x, name=None):
    # return tf.nn.relu(x,name = name)
    with ops.name_scope(name, "my_swish", [x]) as name:

        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return tf.nn.swish(x)
        except AttributeError:
            pass

        return x * tf.nn.sigmoid(x)


def Squeeze_excitation_layer(input_x, out_dim, se_dim, swish, name):
    with tf.name_scope(name):

        squeeze = math_ops.reduce_mean(input_x, [1, 2], name=name + '_gap', keepdims=True)

        excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=se_dim, name=name + '_fully_connected1')

        if swish:
            excitation = swish_activation(excitation, name=name + '_swish')
        else:
            excitation = tf.nn.relu(excitation, name=name + '_relu')

        excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=out_dim, name=name + '_fully_connected2')

        excitation = tf.nn.sigmoid(excitation, name=name + '_sigmoid')

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        scale = input_x * excitation

        return scale


def GroupedConv2D(inputs, filters, kernel, strides, kernel_initializer, padding='same', use_bias=False, name=None):
    channel_axis = -1
    input_channles = inputs.get_shape().as_list()[channel_axis]

    if isinstance(kernel, list) or isinstance(kernel, tuple):
        groups = len(kernel)
        split_inputs = split_channels(input_channles, groups)
        split_filters = split_channels(filters, groups)
        x_splits = tf.split(inputs, split_inputs, channel_axis)
        out = []
        for n in range(groups):
            y = tf.layers.conv2d(x_splits[n], split_filters[n], kernel_size=kernel[n], strides=strides,
                                 name=name + '_GroupedConv' + str(n),
                                 padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer)
            out.append(y)
        out = tf.concat(out, axis=channel_axis)
    else:
        out = tf.layers.conv2d(inputs, filters, kernel_size=kernel, strides=strides, name=name + '_Conv2D',
                               padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer)
    return out


def MDConv(inputs, kernel, strides, kernel_initializer, padding='same', use_bias=False, name=None):
    channel_axis = -1
    input_channles = inputs.get_shape().as_list()[channel_axis]

    if isinstance(kernel, list) or isinstance(kernel, tuple):
        groups = len(kernel)
        split_inputs = split_channels(input_channles, groups)
        x_splits = tf.split(inputs, split_inputs, channel_axis)
        out = []
        for n in range(groups):
            y = tf.keras.layers.DepthwiseConv2D(kernel[n], strides, depth_multiplier=1, padding=padding,
                                                use_bias=use_bias,
                                                kernel_initializer=kernel_initializer)(x_splits[n])
            out.append(y)
        out = tf.concat(out, axis=channel_axis)
    else:
        out = tf.keras.layers.DepthwiseConv2D(kernel, strides, depth_multiplier=1, padding=padding, use_bias=use_bias,
                                              kernel_initializer=kernel_initializer)(inputs)
    return out


def MixNetBlock(inputs, input_filters, output_filters,
                dw_kernel_size, expand_kernel_size,
                project_kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                training=False,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                swish=False, dilated=None,
                data_format=None,
                name=None,
                keep_dropout_backbone=True):
    #    if data_format == 'channels_first':
    #        channel_axis = 1
    #        spatial_dims = [2, 3]
    #    else:
    #        channel_axis = -1
    #        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    if expand_ratio != 1:
        x = GroupedConv2D(inputs, filters, kernel=expand_kernel_size, strides=(1, 1),
                          kernel_initializer=conv_kernel_initializer, name=name + 'GroupConv')

        x = tf.layers.batch_normalization(x, training=training, momentum=batch_norm_momentum,
                                          epsilon=batch_norm_epsilon, name=name + '_bn')
        if swish:
            x = swish_activation(x, name=name + '_swish')
        else:
            x = tf.nn.relu(x, name=name + '_relu')
    else:
        x = inputs

    kernel_size = dw_kernel_size

    x = MDConv(x, kernel_size, strides, kernel_initializer=conv_kernel_initializer,
               padding='same', use_bias=False, name=name + '_MDConv')
    x = tf.layers.batch_normalization(x, training=training, momentum=batch_norm_momentum,
                                      epsilon=batch_norm_epsilon, name=name + '_bn1')
    if swish:
        x = swish_activation(x, name=name + '_swish')
    else:
        x = tf.nn.relu(x, name=name + '_relu1')

    if has_se:
        num_reduced_filters = max(1, int(input_filters * se_ratio))
        x = Squeeze_excitation_layer(x, filters, num_reduced_filters, swish=swish, name=name + 'se_layer')

    x = GroupedConv2D(x, output_filters, kernel=project_kernel_size, strides=(1, 1),
                      kernel_initializer=conv_kernel_initializer, padding='same', use_bias=False,
                      name=name + '_GroupConv2')

    x = tf.layers.batch_normalization(x, training=training, momentum=batch_norm_momentum,
                                      epsilon=batch_norm_epsilon, name=name + '_bn2')

    if id_skip:
        if all(s == 1 for s in strides) and (input_filters == output_filters):
            # only apply drop_connect if skip presents.
            if drop_connect_rate and keep_dropout_backbone:
                x = tf.nn.dropout(x, keep_prob=1 - drop_connect_rate, noise_shape=(None, 1, 1, 1),
                                  name=name + 'dropout')
            x = tf.add(x, inputs, name=name + 'add')
    return x


def MixNet(inputs, input_shape,
           block_args_list: List[BlockArgs],
           depth_multiplier: float,
           include_top=True,
           weights=None,
           pooling=None,
           classes=1000,
           dropout_rate=0.,
           drop_connect_rate=0.,
           training=False,
           batch_norm_momentum=0.99,
           batch_norm_epsilon=1e-3,
           depth_divisor=8,
           stem_size=16,
           feature_size=1536,
           min_depth=None,
           data_format=None,
           default_size=None,
           scope='Mixnet',
           reuse=False,
           keep_dropout_backbone=True):
    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_mixnet_small()

    with tf.variable_scope(scope, reuse=reuse):
        # Stem part
        x = inputs
        x = GroupedConv2D(x, filters=round_filters(stem_size, depth_multiplier, depth_divisor, min_depth), kernel=3,
                          strides=[2, 2], kernel_initializer=conv_kernel_initializer, use_bias=False, name='stem_conv')

        x = tf.layers.batch_normalization(x, training=training, momentum=batch_norm_momentum,
                                          epsilon=batch_norm_epsilon, name='stem_bn')
        x = tf.nn.relu(x, name='stem_relu')

        num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
        drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

        out = []
        for block_idx, block_args in enumerate(block_args_list):
            assert block_args.num_repeat > 0

            # Update block input and output filters based on depth multiplier.
            block_args.input_filters = round_filters(block_args.input_filters, depth_multiplier, depth_divisor,
                                                     min_depth)
            block_args.output_filters = round_filters(block_args.output_filters, depth_multiplier, depth_divisor,
                                                      min_depth)
            block_args.num_repeat = round_repeats(block_args.num_repeat)

            if block_args.strides == [2, 2]:
                out.append(x)
            # The first block needs to take care of stride and filter size increase.
            x = MixNetBlock(x, block_args.input_filters, block_args.output_filters,
                            block_args.dw_kernel_size, block_args.expand_kernel_size,
                            block_args.project_kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            training, batch_norm_momentum, batch_norm_epsilon, block_args.swish,
                            block_args.dilated, data_format, name='MixnetBlock_' + str(block_idx),
                            keep_dropout_backbone=keep_dropout_backbone)

            if block_args.num_repeat > 1:
                block_args.input_filters = block_args.output_filters
                block_args.strides = [1, 1]

            for n in range(block_args.num_repeat - 1):
                x = MixNetBlock(x, block_args.input_filters, block_args.output_filters,
                                block_args.dw_kernel_size, block_args.expand_kernel_size,
                                block_args.project_kernel_size, block_args.strides,
                                block_args.expand_ratio, block_args.se_ratio,
                                block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                                training, batch_norm_momentum, batch_norm_epsilon, block_args.swish,
                                block_args.dilated, data_format,
                                name='MixnetBlock_' + str(block_idx) + '_' + str(n),
                                keep_dropout_backbone=keep_dropout_backbone)

        out.append(x)
        return out


def MixNetSmall(inputs, input_shape=None, include_top=True,
                pooling=None, classes=1000, dropout_rate=0.2, drop_connect_rate=0., training=False,
                data_format=None, scope=None, reuse=False, keep_dropout_backbone=True):
    return MixNet(inputs, input_shape, get_mixnet_small(), depth_multiplier=1.0, include_top=include_top,
                  pooling=pooling,
                  classes=classes, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate, training=training,
                  data_format=data_format, default_size=224, scope=scope, reuse=reuse, keep_dropout_backbone=keep_dropout_backbone)


def MixNetMedium(inputs, input_shape=None, include_top=True,
                 pooling=None, classes=1000, dropout_rate=0.25, drop_connect_rate=0., training=False,
                 data_format=None, scope=None, reuse=False):
    return MixNet(inputs, input_shape, get_mixnet_medium(), depth_multiplier=1.0,
                  include_top=include_top,
                  pooling=pooling,
                  classes=classes, dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate, stem_size=24, traning=training,
                  data_format=data_format, default_size=224, scope=scope, reuse=reuse)


def MixNetLarge(inputs, input_shape=None, include_top=True,
                pooling=None, classes=1000, training=False,
                dropout_rate=0.3, drop_connect_rate=0., data_format=None, scope=None, reuse=False):
    return MixNet(inputs, input_shape, get_mixnet_large(), depth_multiplier=1.3,
                  include_top=include_top, pooling=pooling,
                  classes=classes, dropout_rate=dropout_rate, training=training,
                  drop_connect_rate=drop_connect_rate, stem_size=24,
                  data_format=data_format, default_size=224, scope=scope, reuse=reuse)



