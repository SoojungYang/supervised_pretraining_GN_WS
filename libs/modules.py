import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

from libs.layers import *


# TODO list
# 1. weight standardization
# 2. fix group normalization

class NodeEmbedding(layers.Layer):
    def __init__(self,
                 out_dim,
                 num_heads,
                 use_ffnn,
                 dropout_rate,
                 nm_type='gn',
                 num_groups=8):
        super(NodeEmbedding, self).__init__()

        pre_act = True
        if use_ffnn:
            pre_act = False

        self.gconv = GraphAttn(out_dim, num_heads, pre_act)
        self.use_ffnn = use_ffnn
        if use_ffnn:
            self.ffnn = feed_forward_net(out_dim)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        if nm_type == 'gn':
            # TODO: Fix group norm axis (channels last setup)
            self.norm = tfa.layers.GroupNormalization(groups=num_groups, axis=-1)
        else:
            self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, adj, training):
        h = x
        h = self.gconv(h, adj)

        if self.use_ffnn:
            h = self.ffnn(h)

        h = self.dropout(h, training=training)
        h += x
        h = self.norm(h)

        return h


class FineTuner(layers.Layer):
    def __init__(self,
                 num_heads,
                 out_dim,
                 last_activation,
                 name='output'):
        super(FineTuner, self).__init__()
        self.out_dim = out_dim
        self.readout = PMAReadout(out_dim, num_heads)
        self.dense = layers.Dense(1, input_shape=[out_dim], activation=last_activation, name=name)

    def call(self, x):
        x = self.readout(x)
        z = self.dense(tf.reshape(x, [-1, self.out_dim]))
        return z
