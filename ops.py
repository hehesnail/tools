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
conv2d
"""
def conv2d(x, output_dim, 
        k_h=4, k_w=4, d_h=2, d_w=2, stddev = 0.02,
        name="conv2d", sn=False):
    """
    input:
        x: [B, H, W, C]
        k_h, k_w, d_h, d_w: kernerl size, stride
        sn: True to use spectral normlization
    output:
        x: [B, H/d_h, W/d_w, output_dim]
    """
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k_h, k_w, x.get_shape()[-1], output_dim], 
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            w = sepctral_norm(w)
        conv = tf.nn.conv2d(x, w, strides=[1, d_w, d_w, 1], padding="SAME")


"""
Batch normlization
"""
def batch_norm(x, is_training, momentum=0.9, epsilon=1e-5, name="batch_norm"):
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
        with_att: true to return attention map and gamma
    output:
        [batch_size, H, W, C] after adding attention map
    """
    with tf.variable_scope(name):
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




    
