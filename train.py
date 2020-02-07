import argparse
import os
import sys
import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from absl import app
from absl import logging

from libs.modules import *
from libs.utils import *
from libs.preprocess import *
from libs.dataset import *
from model import *
from args import *


FLAGS = None
np.set_printoptions(3)
tf.random.set_seed(1234)

cmd = set_cuda_visible_device(1)
print("Using ", cmd[:-1], "-th GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = cmd[:-1]

def evaluation_step(model, dataset, multitask_metrics, model_name='', mc_dropout=False, save_outputs=False):

    label_total = np.empty([0, 4])
    pred_total = np.empty([0, 4])

    st = time.time()
    for batch, (x, adj, labels) in enumerate(dataset):

        if mc_dropout:
            preds = [model(x, adj, True) for _ in range(FLAGS.mc_sampling)]
            preds = tf.reduce_mean(preds, axis=1)
        else:
            preds = model(x, adj, False)

        if save_outputs:
            label_total = np.concatenate((label_total, labels.numpy()), axis=0)
            pred_total = np.concatenate((pred_total, preds.numpy()), axis=0)

        for i in range(len(multitask_metrics)):
            for metric in multitask_metrics[i]:
                metric(labels[i], preds[i])

    et = time.time()
    print("Test ", end='')
    for i in range(len(multitask_metrics)):
        print(FLAGS.prop[i], end='')
        for metric in multitask_metrics[i]:
            print(metric.name + ':', metric.result().numpy(), ' ', end='')
            metric.reset_states()
    print("Time:", round(et - st, 3))

    if save_outputs:
        model_name += '_' + str(mc_dropout)
        np.save('./outputs/' + model_name + '_label.npy', label_total)
        np.save('./outputs/' + model_name + '_pred.npy', pred_total)
        return label_total, pred_total

    else:
        return


def train(model, smi):
    model_name = FLAGS.prefix
    model_name += '_' + str(FLAGS.num_embed_layers)
    model_name += '_' + str(FLAGS.embed_dim)
    model_name += '_' + str(FLAGS.finetune_dim)
    model_name += '_' + str(FLAGS.num_embed_heads)
    model_name += '_' + str(FLAGS.num_finetune_heads)
    model_name += '_' + str(FLAGS.embed_dp_rate)
    model_name += '_' + str(FLAGS.prior_length)
    model_name += '_' + str(FLAGS.embed_nm_type)
    ckpt_path = './save/' + model_name + '.ckpt'
    tsbd_path = './log/' + model_name

    num_train = int(len(smi) * 0.8)
    test_smi = smi[num_train:]
    train_smi = smi[:num_train]
    train_ds = get_dataset(train_smi, FLAGS.shuffle_buffer_size, FLAGS.batch_size)
    test_ds = get_dataset(test_smi, FLAGS.shuffle_buffer_size, FLAGS.batch_size)

    step = tf.Variable(0, trainable=False)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[FLAGS.decay_steps, FLAGS.decay_steps * 2],
        values=[1.0, 0.1, 0.01],
    )
    lr = lambda: FLAGS.init_lr * schedule(step)
    coeff = FLAGS.prior_length * (1.0 - FLAGS.embed_dp_rate)
    wd = lambda: coeff * schedule(step)

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

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_loss',
            save_best_only=False,
            save_freq='epoch',
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=tsbd_path,
            histogram_freq=1,
            embeddings_freq=1,
            update_freq='epoch'
        )
    ]

    st_total = time.time()
    print("model compiled and callbacks set")


    history = model.fit(train_ds,
                        epochs=FLAGS.num_epochs,
                        callbacks=callbacks,
                        validation_data=test_ds)
    print('\n', history.history)

    et_total = time.time()
    print("Total time for training:", round(et_total - st_total, 3))

    if FLAGS.save_outputs:
        print("Save the predictions for test dataset")
        evaluation_step(model, test_ds, multitask_metrics, model_name, mc_dropout=False, save_outputs=True)
        evaluation_step(model, test_ds, multitask_metrics, model_name, mc_dropout=True, save_outputs=True)

    return


def main(_):
    def print_model_spec():
        print("Target property", FLAGS.prop)
        print("Random seed for data spliting", FLAGS.seed)
        print("Number of graph convolution layers for node embedding", FLAGS.num_embed_layers)
        print("Dimensionality of node embedding features", FLAGS.embed_dim)
        print("Dimensionality of graph features for fine-tuning", FLAGS.finetune_dim)
        print()
        print("Number of attention heads for node embedding", FLAGS.num_embed_heads)
        print("Number of attention heads for fine-tuning", FLAGS.num_finetune_heads)
        print("Type of normalization", FLAGS.embed_nm_type)
        print("Whether to use feed-forward network", FLAGS.embed_use_ffnn)
        print("Dropout rate", FLAGS.embed_dp_rate)
        print("Weight decay coeff", FLAGS.prior_length)
        print()
        return

    last_activation = []
    for prop in FLAGS.prop:
        if FLAGS.loss_dict[prop] == 'mse':
            last_activation.append(None)
        else:
            last_activation.append(tf.nn.sigmoid)

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
    print_model_spec()

    smi_data = open('./data/smiles_test.txt', 'r')
    smi_data = [line.strip('\n') for line in smi_data.readlines()]
    train(model, smi_data)
    return


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
