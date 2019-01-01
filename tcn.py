import tensorflow as tf

"""
Causal conv1d class with dialated conv
Only need to mask the previous input before time T with all zeros
Directly use tf.layers.Conv1D
"""
class CausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )

    def call(self, inputs):
        #inputs [batch_size, length, channels]
        padding = (self.kernel_size[0]-1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0,0), (1,0), (0,0)])* padding)
        return super(CausalConv1D, self).call(inputs)

"""
Temporal convolutional resblock
The shortcut connection may need projection to have the same shape ot causal conv ops
"""    
class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2,
                trainable=True, name=None, dtype=None,
                activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )

        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=None,
            name="conv1")       
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=None,
            name="conv2")
        self.down_sample = None
    
    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)
    
    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)

        if self.down_sample is not None:
            inputs = self.down_sample(inputs)

        return tf.nn.relu(x + inputs)        

"""
The temporal convnet stacks several temporal conv blocks
The dilation rate of conv op is 2 ** i in order to make the 
receptive field conver the whole input sequence. 
"""
class TemporalConvNet(tf.layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2, 
                trainable=True, name=None, dtype=None,
                activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dype, 
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )

        self.layers = []
        num_levels = len(num_channels) #The layer number of the network
        for i in range(num_levels):
            dilation_rate = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=1, dilation_rate=dilation_rate,
                        dropout=dropout, name="tblock_{}".format(i))
                )
    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs