import tensorflow as tf
from tensorflow.keras import layers

from libs.modules import NodeEmbedding
from libs.modules import FineTuner


class Model(tf.keras.Model):
    def __init__(self,
                 num_props,
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
        self.num_props = num_props

        self.first_embedding = layers.Dense(embed_dim, use_bias=False)
        self.node_embedding = [NodeEmbedding(embed_dim, num_embed_heads,
                                             embed_use_ffnn, embed_dp_rate, embed_nm_type, num_groups)
                               for _ in range(num_embed_layers)]

        self.fine_tuners = []
        for i in range(num_props):
            self.fine_tuners.append(FineTuner(num_finetune_heads, finetune_dim, last_activation[i]))

    def call(self, x, adj, training):
        h = self.first_embedding(x)
        for i in range(self.num_embed_layers):
            h = self.node_embedding[i](h, adj, training)

        outputs = []
        for i in range(self.num_props):
            outputs.append(self.fine_tuners[i](h))
        outputs = tf.transpose(tf.squeeze(outputs))
        return outputs




