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
from model import *


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



def get_dataset(smi):
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
        lambda x: tf.py_function(func=calc_properties,
                                 inp=[x], Tout=[tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y = y.padded_batch(FLAGS.batch_size, padded_shapes=([],[],[],[]))
    ds = tf.data.Dataset.zip((ds, y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


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
    ckpt_path = './save/' + model_name
    tsbd_path = './log/' + model_name

    num_train = int(len(smi) * 0.8)
    test_smi = smi[num_train:]
    train_smi = smi[:num_train]
    train_ds = get_dataset(train_smi)
    test_ds = get_dataset(test_smi)

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
                        epochs=FLAGS.num_epoches,
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

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
