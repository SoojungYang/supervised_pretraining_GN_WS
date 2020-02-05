import tensorflow as tf
from tensorflow.keras import layers


def feed_forward_net(dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(4 * dim, activation=tf.nn.relu),
        tf.keras.layers.Dense(dim)
    ])


# weight standardization regularization
def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)
    return kernel


class GraphConv(layers.Layer):
    def __init__(self, out_dim, pre_act, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(units=out_dim, use_bias=False, kernel_regularizer=ws_reg)
        self.act = tf.nn.relu
        self.pre_act = pre_act

    def call(self, x, adj):
        h = self.dense(x)
        h = tf.matmul(adj, h)
        if self.pre_act:
            h = self.act(h)
        return h


class MultiHeadAttention(layers.Layer):
    def __init__(self, out_dim, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        assert out_dim % num_heads == 0

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.depth = out_dim // num_heads

        self.wq = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.wk = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.wv = tf.keras.layers.Dense(out_dim, use_bias=False)

    def multi_head_attention(self, xq, xk, xv, adj):
        matmul_qk = tf.matmul(xq, xk, transpose_b=True)
        scale = tf.cast(tf.shape(xk)[-1], tf.float32)

        if adj is not None:
            adj = tf.tile(tf.expand_dims(adj, 1), [1, self.num_heads, 1, 1])
        attn = matmul_qk / tf.math.sqrt(scale)
        if adj is not None:
            attn = tf.multiply(attn, adj)
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, xv)
        return out

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, adj=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        h = self.multi_head_attention(q, k, v, adj)
        h = tf.reshape(h, (batch_size, -1, self.out_dim))
        return h


class GraphAttn(layers.Layer):
    def __init__(self, out_dim, num_heads, pre_act, **kwargs):
        super(GraphAttn, self).__init__(**kwargs)

        assert out_dim % num_heads == 0

        self.depth = out_dim // num_heads
        self.num_heads = num_heads
        self.wq = [tf.keras.layers.Dense(units=self.depth, use_bias=False) for _ in range(num_heads)]
        self.wk = [tf.keras.layers.Dense(units=self.depth, use_bias=False) for _ in range(num_heads)]
        self.wv = [tf.keras.layers.Dense(units=self.depth, use_bias=False) for _ in range(num_heads)]
        self.dense = tf.keras.layers.Dense(units=out_dim, use_bias=False, kernel_regularizer=ws_reg)

        self.act = tf.nn.relu
        self.pre_act = pre_act

    def attn_matrix(self, q, k, adj):
        scale = tf.cast(tf.shape(k)[-1], tf.float32)
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.multiply(attn, adj)
        attn /= tf.math.sqrt(scale)

        attn = tf.nn.tanh(attn)

        return attn

    def call(self, x, adj):
        h_list = []
        for i in range(self.num_heads):
            q = self.wq[i](x)
            k = self.wk[i](x)
            v = self.wv[i](x)
            attn = self.attn_matrix(q, k, adj)
            h = tf.matmul(attn, v)
            h_list.append(h)
        h = tf.concat(h_list, axis=-1)
        h = self.dense(h)
        if self.pre_act:
            h = self.act(h)
        return h


class MeanPooling(layers.Layer):
    def __init__(self, axis, **kwargs):
        super(MeanPooling, self).__init__()
        self.axis = axis

    def call(self, x):
        return tf.math.reduce_mean(x, axis=self.axis)


class SumPooling(layers.Layer):
    def __init__(self, axis, **kwargs):
        super(SumPooling, self).__init__()
        self.axis = axis

    def call(self, x):
        return tf.math.reduce_sum(x, axis=self.axis)


class MaxPooling(layers.Layer):
    def __init__(self, axis, **kwargs):
        super(MaxPooling, self).__init__()
        self.axis = axis

    def call(self, x):
        return tf.math.reduce_max(x, axis=self.axis)


class LinearReadout(layers.Layer):
    def __init__(self, out_dim, pooling, **kwargs):
        super(LinearReadout, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(units=out_dim, use_bias=False)
        self.act = tf.nn.sigmoid
        self.pooling = None
        if pooling == 'mean':
            self.pooling = MeanPooling(axis=1)
        elif pooling == 'sum':
            self.pooling = SumPooling(axis=1)
        elif pooling == 'max':
            self.pooling = MaxPooling(axis=1)

    def call(self, x):
        z = self.dense(x)
        z = self.pooling(z)
        return self.act(z)


class GraphGatherReadout(layers.Layer):
    def __init__(self, out_dim, pooling='mean', **kwargs):
        super(GraphGatherReadout, self).__init__(**kwargs)

        self.dense1 = tf.keras.layers.Dense(
            units=out_dim,
            use_bias=False,
            activation=tf.nn.sigmoid
        )
        self.dense2 = tf.keras.layers.Dense(
            units=out_dim,
            use_bias=False
        )
        self.act = tf.nn.sigmoid
        self.pooling = None
        if pooling == 'mean':
            self.pooling = MeanPooling(axis=1)
        elif pooling == 'sum':
            self.pooling = SumPooling(axis=1)
        elif pooling == 'max':
            self.pooling = MaxPooling(axis=1)

    def call(self, x):
        z = tf.multiply(self.dense1(x), self.dense2(x))
        z = self.pooling(z)
        return self.act(z)


class PMAReadout(layers.Layer):
    def __init__(self, out_dim, num_heads, num_seeds=1, **kwargs):
        super(PMAReadout, self).__init__(**kwargs)

        self.out_dim = out_dim
        self.num_heads = num_heads

        init = tf.initializers.glorot_normal()
        self.seed_vector = tf.Variable(
            initial_value=tf.ones(shape=(1, num_seeds, 64),
                                  dtype='float32'),
            trainable=False
        )
        self.mha = MultiHeadAttention(out_dim, num_heads)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.act = tf.nn.sigmoid

    def call(self, x):
        batch_size = tf.shape(x)[0]
        num_nodes = tf.cast(tf.shape(x)[1], tf.float32)
        out = tf.tile(self.seed_vector, [batch_size, 1, 1])
        out = self.mha(out, x, x)
        out = tf.squeeze(out) * num_nodes
        return self.act(out)
