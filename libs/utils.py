import os

import tensorflow as tf
import tensorflow_addons as tfa

from libs.lr_scheduler import WarmUpSchedule


def set_cuda_visible_device(ngpus):
    empty = []
    for i in range(4):
        os.system('nvidia-smi -i ' + str(i) + ' | grep "No running" | wc -l > empty_gpu_check')
        f = open('empty_gpu_check')
        out = int(f.read())
        if int(out) == 1:
            empty.append(i)
    if len(empty) < ngpus:
        print('avaliable gpus are less than required')
    cmd = ''
    for i in range(ngpus):
        cmd += str(empty[i]) + ','
    return cmd


def get_learning_rate_scheduler(lr_schedule='stair',
                                graph_dim=256,
                                warmup_steps=1000,
                                init_lr=1e-3,
                                decay_steps=500,
                                decay_rate=0.1,
                                staircase=True):
    scheduler = None
    if lr_schedule == 'warmup':
        scheduler = WarmUpSchedule(
            d_model=graph_dim,
            warmup_steps=warmup_steps
        )

    elif lr_schedule == 'stair':

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=init_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )

    return scheduler


def get_loss_function(loss_type):
    loss_fn = None
    if loss_type == 'bce':
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    elif loss_type == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        if 'focal' in loss_type:
            alpha = float(loss_type.split('_')[1])
            gamma = float(loss_type.split('_')[2])
            loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
                from_logits=False, alpha=alpha, gamma=gamma
            )
    return loss_fn


class MultitaskLoss(tf.keras.losses.Loss):
    def __init__(self,
                 prop_list,
                 loss_dict):
        super(MultitaskLoss, self).__init__()
        self.num_props = len(prop_list)
        self.loss_fn_list = []
        for prop in prop_list:
            self.loss_fn_list.append(get_loss_function(loss_dict[prop]))

    def call(self, y_true, y_pred):
        loss = 0
        for i in range(self.num_props):
            loss += self.loss_fn_list[i](y_true[:, i], y_pred[:, i])
        return loss


def get_metric_list(loss_type):

    if loss_type in ['mse']:
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name='MAE'),
            tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
        ]

    else:
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
            tf.keras.metrics.AUC(curve='ROC', name='AUROC'),
            tf.keras.metrics.AUC(curve='PR', name='AUPRC'),
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.Recall(name='Recall'),
        ]

    return metrics
