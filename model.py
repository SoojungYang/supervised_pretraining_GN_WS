import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from libs.modules import NodeEmbedding
from libs.modules import FineTuner


class Model(tf.keras.Model):
    def __init__(self,
                 list_props,
                 num_embed_layers,
                 embed_dim,
                 finetune_dim,
                 num_embed_heads=4,
                 num_finetune_heads=4,
                 embed_use_ffnn=True,
                 embed_dp_rate=0.1,
                 embed_nm_type='gn',
                 num_groups=8,
                 last_activation=None):
        super(Model, self).__init__()

        self.num_embed_layers = num_embed_layers
        self.num_props = len(list_props)

        self.first_embedding = layers.Dense(embed_dim, use_bias=False)
        self.node_embedding = [NodeEmbedding(embed_dim, num_embed_heads,
                                             embed_use_ffnn, embed_dp_rate, embed_nm_type, num_groups)
                               for _ in range(num_embed_layers)]

        self.fine_tuners = []
        for i in range(self.num_props):
            self.fine_tuners.append(FineTuner(num_finetune_heads, finetune_dim, last_activation[i], name=list_props[i]))

    def call(self, data, training):
        x = data['x']
        adj = data['a']
        h = self.first_embedding(x)
        for i in range(self.num_embed_layers):
            h = self.node_embedding[i](h, adj, training)
        outputs = []
        for i in range(self.num_props):
            output = tf.squeeze(self.fine_tuners[i](h))
            outputs.append(tf.reshape(output, [-1]))
        return outputs[0], outputs[1], outputs[2], outputs[3]
