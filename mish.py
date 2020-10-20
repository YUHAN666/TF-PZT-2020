""" implementation of mish activation function """
import tensorflow as tf
from tensorflow.python.framework import ops


def mish(input_tensor, name):
    with ops.name_scope(name, "my_mish", [input_tensor]) as name:
        return input_tensor * tf.math.tanh(tf.math.softplus(input_tensor))