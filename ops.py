import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

"""
conv2d for downsampling
Using He initialization
"""
def conv2d(x, output_dim,
        k_h=4, k_w=4, d_h=2, d_w=2, stddev = 0.02,
        name="conv2d", sn=False, with_w=False, padding="SAME"):
    """
    input:
        x: [N, H, W, C]
        k_h, k_w, d_h, d_w: kernerl size, stride
        sn: True to use spectral normlization
        with_w: True to return weights and biases
        padding: default "SAME"
    output:
        x: [N, H/d_h, W/d_w, output_dim]
    """
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [k_h, k_w, x.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            w = sepctral_norm(w)
        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable("biases", [output_dim],
                                initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        if with_w:
            return conv, w, biases
        else:
            return conv

"""
conv3d for downsampling
Using He initializetion
"""
def conv3d(x, output_dim,
        k_d=4, k_h=4, k_w=4, d_d=2, d_h=2, d_w=2, stddev = 0.02,
        name="conv3d", with_w=False, padding="SAME"):
    """
    input:
        x: [N, D, H, W, C]
        k_d, k_h, k_w, d_d, d_h, d_w: kernerl size, strides
        with_w: True to return weights and biases
        padding: default "SAME"
    output:
        x: [N, D/d_d, H/d_h, W/d_w, output_dim]
    """
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [k_d, k_h, k_w, x.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(x, w, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable("biases", [output_dim],
                                initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        if with_w:
            return conv, w, biases
        else:
            return conv

"""
deconv2d for upsampling
Using He initialization
"""
def deconv2d(x, output_shape,
            k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
            name="deconv2d", sn=False, with_w=False, padding="SAME"):
    """
    input:
        x:[N, H, W, C]
        output_shape: [N, H', W', C']
        k_h, k_w, d_h, d_w: kernel size and strides
        sn: True to use spectral normalization
        with_w: True to return weights and biases
    output:
        deconv feature map with shape as output_shape
    """
    with tf.variable_scope(name) as scope:
        #filter: [k_h, k_w, out_c, in_c]
        w = tf.get_variable("w", [k_h, k_w, output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            w = sepctral_norm(w)
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape,
                            strides=[1,d_h,d_w,1], padding=padding)
        biases = tf.get_variable("biases", [output_shape[-1]],
                                initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

"""
deconv3d for upsampling
Using He initialization
"""
def deconv3d(x, output_shape,
            k_d=4, k_h=4, k_w=4, d_d=2, d_h=2, d_w=2, stddev=0.02,
            name="deconv3d", with_w=False, padding="SAME"):
    """
    input:
        x:[N, D, H, W, C]
        output_shape: [N, D', H', W', C']
        k_d, k_h, k_w, d_d, d_h, d_w: kernel size and strides
        with_w: True to return weights and biases
    output:
        deconv feature map with shape as output_shape
    """
    with tf.variable_scope(name) as scope:
        #filter: [k_d, k_h, k_w, out_c, in_c]
        w = tf.get_variable("w", [k_d, k_h, k_w, output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv3d_transpose(x, w, output_shape=output_shape,
                            strides=[1,d_d,d_h,d_w,1], padding=padding)
        biases = tf.get_variable("biases", [output_shape[-1]],
                                initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

"""
fully connected layer for projecting
"""
def linear(x, output_size, name="linear",
        stddev=0.02, bias_start=0.0, sn=False, with_w=False):
    """
    input:
        x: [N, D]
        sn: True to use spectral normalization
        with_w: True to return weights and biases
    output:
        [N, output_size]
    """
    with tf.variable_scope(name) as scope:
        shape = x.get_shape().as_list()
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable("biases", [output_size],
                            initializer=tf.constant_initializer(bias_start))
        if sn:
            w = sepctral_norm(w)
        fc = tf.matmul(x, w) + biases

        if with_w:
            return fc, w, biases
        else:
            return fc

"""
Batch normlization
"""
def batch_norm(x, is_training=False, momentum=0.9, epsilon=1e-5, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        is_training=is_training,
                                        scope=name)
"""
Layer normalization
"""
def layer_norm(x, is_training, name="layer_norm"):
    return tf.contrib.layers.layer_norm(x,
                                        center=True,
                                        scale=True,
                                        trainable=is_training,
                                        scope=name)

"""
L2 norm
"""
def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v**2)**0.5 + eps)

"""
sepctral normalization
"""
def sepctral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.shape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

"""
hw flatten
"""
def hw_flatten(x):
    """
    input:
        x: feature map [B, H, W, C]
    output:
        reshaped feature map [B, H*W, C]
    """
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

"""
self attention layer
"""
def attention_layer(x, name="attention", sn=False, with_att=False):
    """
    input:
        x: [batch_size, H, W, C]
        sn: use spectral normalization or not
        with_att: True to return attention map and gamma
    output:
        [batch_size, H, W, C] after adding attention map
    """
    with tf.variable_scope(name) as scope:
        c = x.get_shape()[-1]
        c_new = c // 8
        f = conv2d(x, c_new, k_h=1, k_w=1, d_h=1, d_w=1, name=name+"f", sn=sn)
        g = conv2d(x, c_new, k_h=1, k_w=1, d_h=1, d_w=1, name=name+"g", sn=sn)
        h = conv2d(x, c, k_h=1, k_w=1, d_h=1, d_w=1, name=name+"h", sn=sn)
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)

        atten = tf.nn.softmax(s)
        o = tf.matmul(atten, hw_flatten(h))
        gamma = tf.get_variable("gamma", shape=[1],initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=x.get_shape())
        x = gamma*o + x

        if with_att:
            return x, atten, gamma
        else:
            return x

"""
leaky relu activation function
"""
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
