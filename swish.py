"""
Swish activation function: x * sigmoid(x).
Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
"""
from tensorflow.python.framework import ops
import tensorflow as tf


def swish(x, name=None):
    # return tf.nn.relu(x,name = name)
    with ops.name_scope(name, "my_swish", [x]) as name:

        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return tf.nn.swish(x)
        except AttributeError:
            pass

        return x * tf.nn.sigmoid(x)
