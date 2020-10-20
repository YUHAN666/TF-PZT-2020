import tensorflow as tf
from config import CLASS_NUM, IMAGE_SIZE, DROP_OUT
import numpy as np


def bn_layer(x, is_training, scope, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer
    Args:
       x: input tensor
       scope: scope name
       is_training: python boolean value
       epsilon: the variance epsilon - a small float number to avoid dividing by 0
       decay: the moving average decay
       Returns: The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable(scope + "_gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable(scope + "_beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable(scope + "_moving_mean", shape[-1], initializer=tf.constant_initializer(0.0),
                                     trainable=False)
        moving_var = tf.get_variable(scope + "_moving_variance", shape[-1], initializer=tf.constant_initializer(1.0),
                                     trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
            avg = tf.reshape(avg, [avg.shape.as_list()[-1]])
            var = tf.reshape(var, [var.shape.as_list()[-1]])
            # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_avg = tf.assign(moving_avg, moving_avg * decay + avg * (1 - decay))
            # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            update_moving_var = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
    return output


def bn_layer_top(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    # assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(
        is_training,
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )


def conv_block(inputs, conv_type, filters, kernel_size, strides, training, name=None, padding='same', relu=True,
               use_bias=False):
    if (conv_type == 'ds'):
        x = tf.layers.separable_conv2d(inputs, filters, kernel_size, strides, name=name + '_ds_conv', padding=padding,
                                       use_bias=use_bias)
    else:
        x = tf.layers.conv2d(inputs, filters, kernel_size, strides, name=name + '_conv2d', padding='same',
                             use_bias=use_bias)
    x = tf.layers.batch_normalization(x, training=training, name=name + '_bn')
    # x = bn_layer(x,is_training = training,scope = name+'_bn')

    if (relu):
        x = tf.nn.relu(x, name=name + '_relu')
    return x


def _res_bottleneck(inputs, filters, kernel, t, s, training, name=None, res=False):
    tchannel = inputs.get_shape().as_list()[-1] * t
    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1), name=name + '_conv', training=training)

    x = tf.keras.layers.DepthwiseConv2D(kernel, s, depth_multiplier=1, padding='same')(x)
    x = tf.layers.batch_normalization(x, training=training, name=name + '_bn')
    # x = bn_layer(x,is_training = training,scope = name+'_bn')
    x = tf.nn.relu(x, name=name + 'relu')
    x = conv_block(x, 'conv', filters, (1, 1), strides=1, training=training, name=name + '_conv2', padding='same',
                   relu=False, use_bias=False)
    if DROP_OUT:
        x = tf.nn.dropout(x, keep_prob=0.5, noise_shape=(None, 1, 1, None), name='dropout')
    if res:
        x = x + inputs
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n, training, name=None):
    x = _res_bottleneck(inputs, filters, kernel, t, strides, training, name=name + '_res')
    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, training, name=name + str(i) + '_res', res=True)
    return x


def pyramid_pooling_block(input_tensor, bin_sizes, training, name=None):
    concat_list = [input_tensor]
    w = IMAGE_SIZE[1] // 16
    h = IMAGE_SIZE[0] // 16
    n = 0
    for bin_size in bin_sizes:
        n = n + 1
        x = tf.layers.average_pooling2d(input_tensor, pool_size=(
        h - (bin_size - 1) * (h // bin_size), w - (bin_size - 1) * (w // bin_size)),
                                        strides=(h // bin_size, w // bin_size), name=name + '_' + str(n) + '_agp2d')
        x = conv_block(x, 'conv', 128 // 4, (1, 1), strides=(1, 1), padding='valid', name=name + '_' + str(n) + 'conv',
                       training=training)
        x = tf.image.resize_images(x, (h, w), align_corners=True)
        concat_list.append(x)
    x = tf.concat(concat_list, axis=3)
    x = conv_block(x, 'conv', 128, (1, 1), strides=(1, 1), training=training, name=name + 'conv', padding='valid')
    return x


def LearningToDownsample(inputs, training, scope='lds', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lds_layer = conv_block(inputs, 'conv', 32, (3, 3), strides=(2, 2), training=training, name='conv1',
                               padding='valid')
        lds_layer1 = conv_block(lds_layer, 'ds', 48, (3, 3), strides=(1, 1), training=training, name='dsconv2')
        lds_layer = conv_block(lds_layer1, 'ds', 48, (3, 3), strides=(2, 2), training=training, name='dsconv3')
        return lds_layer, lds_layer1


def GlobalFeatureExtractor(inputs, training, scope='gfe', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        x = bottleneck_block(inputs, 64, (3, 3), 6, (2, 2), n=3, training=training, name='btnk1')
        x = bottleneck_block(x, 96, (3, 3), 6, (2, 2), n=3, training=training, name='btnk2')
        x = bottleneck_block(x, 128, (3, 3), 6, (1, 1), n=3, training=training, name='btnk3')
        x = pyramid_pooling_block(x, [1, 2, 4], training=training, name='ppb')
        return x


def FeatureFusion(lds1, higher_res_feature, low_res_feature, training, scope='ff', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        ff_layer1 = conv_block(higher_res_feature, 'conv', 128, (1, 1), strides=(1, 1), training=training, name='conv',
                               padding='same', relu=False)

        # 与原版不一样，把上采样放到了最后，性能验证无差别
        ff_layer2 = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same')(
            low_res_feature)
        ff_layer2 = tf.layers.batch_normalization(ff_layer2, training=training, name='bn')
        # ff_layer2 = bn_layer(ff_layer2,is_training = training,scope = 'bn')
        ff_layer2 = tf.nn.relu(ff_layer2, name='relu1')
        ff_layer2 = tf.layers.conv2d(ff_layer2, 128, (1, 1), (1, 1), padding='same', name='conv2d')
        ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(ff_layer2)
        x = ff_layer1 + ff_layer2

        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        lds1 = conv_block(lds1, 'conv', 128, (1, 1), strides=(1, 1), training=training, name='conv2', padding='same',
                          relu=False)

        x = x + lds1
        x = tf.nn.relu(x, name='relu2')
        return x


def Classifier(inputs, training, scope='classifier', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        x = conv_block(inputs, 'ds', 128, (3, 3), strides=(1, 1), training=training, name='dsconv1', padding='same')
        features = conv_block(x, 'ds', 128, (3, 3), strides=(1, 1), training=training, name='dsconv2', padding='same')
        if DROP_OUT:
            features = tf.nn.dropout(features, name='dropout')
        logits = conv_block(features, 'conv', CLASS_NUM, (1, 1), strides=(1, 1), training=training, name='conv3',
                            padding='same', relu=False)
        logits = tf.keras.layers.UpSampling2D((2, 2))(logits)
        return features, logits