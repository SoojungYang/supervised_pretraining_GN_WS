import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from absl import app
from absl import logging

from libs.modules import *
from libs.utils import *
from libs.preprocess import *
from libs.dataset import *
from model import *
from args import *


class BenchmarkModel(keras.Model):
    def __init__(self,
                 model,
                 fine_tune_at=3,
                 last_activation=None):
        super(BenchmarkModel, self).__init__()

        self.pre_trained = model.layers[:5]
        self.prediction = keras.layers.Dense(1, activation=last_activation)

        for layer in self.pre_trained[:fine_tune_at]:
            layer.trainable = False

    def call(self, data):
        x, adj = data['x'], data['a']
        h = self.pre_trained[0](x)
        for i in range(1, 5):
            h = self.pre_trained[i](h, adj)
        outputs = self.last_activation(self.prediction(h))
        return outputs


def benchmark(_):

    smi = open('./data/smiles_test.txt', 'r')
    smi = [line.strip('\n') for line in smi.readlines()]
    test_smi = smi[:128]
    benchmark_ds = get_benchmark_dataset(test_smi,
                                         logP_benchmark,
                                         FLAGS.shuffle_buffer_size,
                                         FLAGS.batch_size)

    model = define_model()
    model.load_weights(FLAGS.ckpt_path)
    print("loaded weights of pre-trained model")
    print(model.layers)

    benchmark_last_activation, loss, metrics = get_task_options(FLAGS.benchmark_task_type)

    # Transfer pre-trained weights into Benchmark Model
    model = BenchmarkModel(model, FLAGS.fine_tune_at, benchmark_last_activation)
    print("Stacked Encoder and prediction layer")
    print(model.layers)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    model.fit(benchmark_ds,
              epochs=1)

    model.evaluate(benchmark_ds)
    print(model.summary())

    return


def define_model():
    last_activation = []
    for prop in FLAGS.prop:
        if FLAGS.loss_dict[prop] == 'mse':
            last_activation.append(None)
        else:
            last_activation.append(tf.nn.sigmoid)

    step = tf.Variable(0, trainable=False)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[FLAGS.decay_steps, FLAGS.decay_steps * 2],
        values=[1.0, 0.1, 0.01],
    )
    lr = lambda: FLAGS.init_lr * schedule(step)
    coeff = FLAGS.prior_length * (1.0 - FLAGS.embed_dp_rate)
    wd = lambda: coeff * schedule(step)

    model = Model(
        list_props=FLAGS.prop,
        num_embed_layers=FLAGS.num_embed_layers,
        embed_dim=FLAGS.embed_dim,
        finetune_dim=FLAGS.finetune_dim,
        num_embed_heads=FLAGS.num_embed_heads,
        num_finetune_heads=FLAGS.num_finetune_heads,
        embed_use_ffnn=FLAGS.embed_use_ffnn,
        embed_dp_rate=FLAGS.embed_dp_rate,
        embed_nm_type=FLAGS.embed_nm_type,
        num_groups=FLAGS.num_groups,
        last_activation=last_activation
    )

    optimizer = tfa.optimizers.AdamW(
        weight_decay=wd,
        learning_rate=lr,
        beta_1=FLAGS.beta_1,
        beta_2=FLAGS.beta_2,
        epsilon=FLAGS.opt_epsilon
    )

    # logP, TPSA, MR, MW
    model.compile(optimizer=optimizer,
                  loss={'output_1': keras.losses.MeanSquaredError(),
                        'output_2': keras.losses.MeanSquaredError(),
                        'output_3': keras.losses.MeanSquaredError(),
                        'output_4': keras.losses.MeanSquaredError()},
                  metrics={'output_1': keras.metrics.MeanSquaredError(),
                           'output_2': keras.metrics.MeanSquaredError(),
                           'output_3': keras.metrics.MeanSquaredError(),
                           'output_4': keras.metrics.MeanSquaredError()},
                  loss_weights={'output_1': 2., 'output_2': 2., 'output_3': 2., 'output_4': 1.}
                  )

    return model


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=benchmark, argv=[sys.argv[0]] + unparsed)


