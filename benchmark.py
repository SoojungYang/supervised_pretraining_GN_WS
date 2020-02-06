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
from model import *


class BenchmarkModel(tf.keras.Model):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self.pre_trained = model.layers[:5]
        self.prediction = keras.layers.Dense(1)

    def call(self, data):
        x, adj = data['x'], data['a']
        h = self.pre_trained[0](x)
        for i in range(1, 5):
            h = self.pre_trained[i](h, adj)
        outputs = self.prediction(h)
        return outputs


def benchmark(_):

    smi = open('./data/smiles_test.txt', 'r')
    smi = [line.strip('\n') for line in smi.readlines()]
    test_smi = smi[:128]
    benchmark_ds = get_benchmark_dataset(test_smi, logP_benchmark)

    model = define_model()
    model.load_weights(cp_path)
    print("loaded weight")

    print(model.layers)

    t_model = BenchmarkModel(model)
    print("Stacked Encoder and prediction layer")
    print(t_model.layers)

    # Freeze model
    print("Number of layers in the transferred model: ", len(t_model.layers))
    fine_tune_at = 3
    for layer in t_model.layers[:fine_tune_at]:
        layer.trainable = False

    t_model.compile(loss=keras.losses.MeanSquaredError(),
                    optimizer=optimizer,
                    metrics=[keras.metrics.MeanSquaredError()])

    t_model.fit(benchmark_ds,
                epochs=1)

    t_model.evaluate(benchmark_ds)

    print(t_model.summary())

    return


def get_benchmark_dataset(smi, property_func):
    def x_to_dict(x, a):
        return {'x': x, 'a': a}

    smi = tf.data.Dataset.from_tensor_slices(smi)
    smi = smi.shuffle(FLAGS.shuffle_buffer_size)
    ds = smi.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph,
                                 inp=[x],
                                 Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(FLAGS.batch_size, padded_shapes=([None, 58], [None, None]))
    ds = ds.map(x_to_dict)

    y = smi.map(
        lambda x: tf.py_function(func=property_func,
                                 inp=[x], Tout=tf.float32),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y = y.padded_batch(FLAGS.batch_size, padded_shapes=([]))
    ds = tf.data.Dataset.zip((ds, y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


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

    cp_path = FLAGS.ckpt_path
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
    parser = argparse.ArgumentParser()


    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeEror('Boolean value expected')

    # Hyper-parameters for prefix, prop and random seed
    parser.add_argument('--prefix', type=str, default='test',
                        help='Prefix for this training')
    parser.add_argument('--prop', type=str, default=['logP', 'TPSA', 'MW', 'MR'],
                        help='Target properties to train')
    parser.add_argument('--seed', type=int, default=1111,
                        help='Random seed will be used to shuffle dataset')

    # Hyper-parameters for model construction
    parser.add_argument('--num_embed_layers', type=int, default=4,
                        help='Number of node embedding layers')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Dimension of node embeddings')
    parser.add_argument('--finetune_dim', type=int, default=256,
                        help='Dimension of a fine-tuned z')
    parser.add_argument('--num_embed_heads', type=int, default=4,
                        help='Number of attention heads for node embedding')
    parser.add_argument('--num_finetune_heads', type=int, default=4,
                        help='Number of attention heads for fine-tuning layer')
    parser.add_argument('--embed_use_ffnn', type=str2bool, default=False,
                        help='Whether to use feed-forward nets for node embedding')
    parser.add_argument('--embed_dp_rate', type=float, default=0.1,
                        help='Dropout rates in node embedding layers')
    parser.add_argument("--embed_nm_type", type=str, default='gn',
                        help='Type of normalization: gn or ln')
    parser.add_argument("--num_groups", type=int, default=8,
                        help='Number of groups for group normalization')
    parser.add_argument('--prior_length', type=float, default=1e-4,
                        help='Weight decay coefficient')

    # Hyper-parameters for data loading
    parser.add_argument('--shuffle_buffer_size', type=int, default=100,
                        help='shuffle buffer size')


    # Hyper-parameters for loss function
    parser.add_argument("--loss_dict", type=dict, default={'logP': 'mse', 'TPSA': 'mse',
                                                           'MR': 'mse', 'MW': 'mse', 'SAS': 'mse'},
                        help='type of loss for each property, Options: bce, mse, focal, class_balanced, max_margin')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha in Focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma in Focal loss')

    # Hyper-parameters for training
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_epoches', type=int, default=50,
                        help='Number of epoches')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='Initial learning rate,\
                              Do not need for warmup scheduling')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='Beta1 in adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='Beta2 in adam optimizer')
    parser.add_argument('--opt_epsilon', type=float, default=1e-7,
                        help='Epsilon in adam optimizer')
    parser.add_argument('--decay_steps', type=int, default=40,
                        help='Decay steps for stair learning rate scheduling')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='Decay rate for stair learning rate scheduling')
    parser.add_argument('--max_to_keep', type=int, default=5,
                        help='Maximum number of checkpoint files to be kept')

    # Hyper-parameters for evaluation
    parser.add_argument("--save_outputs", type=str2bool, default=True,
                        help='Whether to save final predictions for test dataset')
    parser.add_argument('--mc_dropout', type=str2bool, default=False,
                        help='Whether to infer predictive distributions with MC-dropout')
    parser.add_argument('--mc_sampling', type=int, default=30,
                        help='Number of MC sampling')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k instances for evaluating Precision or Recall')

    parser.add_argument('--ckpt_path', type=str, default='./save/test01_4_64_256_4_4_0.1_0.0001_gn.ckpt',
                        help='checkpoint file')
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=benchmark, argv=[sys.argv[0]] + unparsed)


